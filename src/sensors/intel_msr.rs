//! Intel per-core and package telemetry from read-only MSRs.
//!
//! Adds what hwmon and cpufreq do not expose: effective and busy clocks and
//! C0 residency from APERF/MPERF/TSC deltas, the requested core and uncore
//! VID, the core and ring ratios, TjMAX and the per-core distance to it, the
//! thermal status flags, the VR current limit, and the raw perf-limit-reason
//! words. Temperatures themselves are left to `coretemp`.
//!
//! All reads go through the allowlisted, read-only
//! [`MsrDevice`](crate::platform::msr::MsrDevice). Reading `/dev/cpu/N/msr`
//! runs RDMSR on CPU N through an IPI, which wakes that CPU: instantaneous
//! fields (VID, ratio) of an idle core describe the operating point it wakes
//! into. Counter-derived values (clocks, residency) are interval averages and
//! are not skewed by this.
//!
//! VID values are the CPU's *requested* voltage (SVID request), not a
//! measured rail. Board-sensed Vcore comes from the Super I/O via hwmon.
//!
//! Needs the `msr` module and root or `CAP_SYS_RAWIO`; without them the
//! source is silently empty.

use crate::model::sensor::{SensorId, SensorReading};

#[cfg(target_arch = "x86_64")]
mod inner {
    use std::path::Path;
    use std::time::Instant;

    use crate::model::sensor::{SensorCategory, SensorId, SensorReading, SensorUnit};
    use crate::platform::msr::{self, MsrDevice, bits};
    use crate::platform::sysfs;

    /// Model-specific MSR fields that the SDM marks reserved, enabled only on
    /// parts where they have been verified (same set as the upstream
    /// turbostat CoreVID/UncVID proposal).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub(super) struct ModelFeatures {
        /// `IA32_PERF_STATUS[47:32]` is the requested core VID (1/8192 V).
        pub core_vid: bool,
        /// `MSR_UNCORE_PERF_STATUS[31:16]` is the requested uncore VID.
        pub uncore_vid: bool,
    }

    pub(super) fn model_features(family: u32, model: u32) -> ModelFeatures {
        if family != 6 {
            return ModelFeatures::default();
        }
        match model {
            // Raptor Lake, Raptor Lake-P, Raptor Lake-S
            0xB7 | 0xBA | 0xBF => ModelFeatures {
                core_vid: true,
                uncore_vid: false,
            },
            // Arrow Lake-S
            0xC6 => ModelFeatures {
                core_vid: true,
                uncore_vid: true,
            },
            _ => ModelFeatures::default(),
        }
    }

    // ── Pure decoders ───────────────────────────────────────────────────

    /// Requested core VID from `IA32_PERF_STATUS`, or `None` when zero.
    pub(super) fn core_vid_volts(perf_status: u64) -> Option<f64> {
        vid_volts(bits(perf_status, 47, 32))
    }

    /// Current core ratio from `IA32_PERF_STATUS[15:8]`, or `None` when zero.
    pub(super) fn core_ratio(perf_status: u64) -> Option<u64> {
        Some(bits(perf_status, 15, 8)).filter(|&r| r != 0)
    }

    /// Requested uncore VID from `MSR_UNCORE_PERF_STATUS[31:16]`.
    pub(super) fn uncore_vid_volts(uncore_status: u64) -> Option<f64> {
        vid_volts(bits(uncore_status, 31, 16))
    }

    /// Current ring (uncore) ratio from `MSR_UNCORE_PERF_STATUS[6:0]`.
    pub(super) fn ring_ratio(uncore_status: u64) -> Option<u64> {
        Some(bits(uncore_status, 6, 0)).filter(|&r| r != 0)
    }

    fn vid_volts(code: u64) -> Option<f64> {
        (code != 0).then(|| code as f64 / 8192.0)
    }

    /// TjMAX in °C from `MSR_TEMPERATURE_TARGET[23:16]`.
    pub(super) fn tjmax(temp_target: u64) -> Option<u64> {
        Some(bits(temp_target, 23, 16)).filter(|&t| t != 0)
    }

    /// TCC activation offset in °C from `MSR_TEMPERATURE_TARGET[29:24]`.
    pub(super) fn tcc_offset(temp_target: u64) -> u64 {
        bits(temp_target, 29, 24)
    }

    /// Degrees below TjMAX from a thermal status word, if the reading is
    /// valid (bit 31).
    pub(super) fn distance_to_tjmax(therm_status: u64) -> Option<u64> {
        (bits(therm_status, 31, 31) == 1).then(|| bits(therm_status, 22, 16))
    }

    /// Status and sticky log flags (bits 15:0) of a thermal status word,
    /// displayed raw and never cleared.
    pub(super) fn therm_flags(therm_status: u64) -> u64 {
        bits(therm_status, 15, 0)
    }

    /// VR current limit in amps from `MSR_VR_CURRENT_CONFIG[12:0]`
    /// (1/8 A units). This is a current, not a power limit.
    pub(super) fn vr_current_limit_amps(vr_config: u64) -> Option<f64> {
        let raw = bits(vr_config, 12, 0);
        (raw != 0).then(|| raw as f64 / 8.0)
    }

    /// Ring ratio limits `(min, max)` from `MSR_RING_RATIO_LIMIT`
    /// (`[14:8]` min, `[6:0]` max), the same fields intel-uncore-frequency uses.
    pub(super) fn ring_ratio_limits(raw: u64) -> Option<(u64, u64)> {
        let (min, max) = (bits(raw, 14, 8), bits(raw, 6, 0));
        (max != 0).then_some((min, max))
    }

    /// Residency percentage of a counter that ticks at the TSC rate.
    pub(super) fn residency_pct(prev: (u64, u64), cur: (u64, u64)) -> Option<f64> {
        let d_tsc = cur.0.checked_sub(prev.0)?;
        let d_res = cur.1.checked_sub(prev.1)?;
        (d_tsc != 0).then(|| (d_res as f64 / d_tsc as f64 * 100.0).min(100.0))
    }

    /// APERF/MPERF/TSC sample of one logical CPU.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct Counters {
        pub tsc: u64,
        pub aperf: u64,
        pub mperf: u64,
        pub at: Instant,
    }

    /// Interval averages derived from two counter samples.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(super) struct ClockStats {
        /// Average clock including idle time (ΔAPERF / Δt).
        pub effective_mhz: f64,
        /// C0 residency (ΔMPERF / ΔTSC).
        pub busy_pct: f64,
        /// Average clock while in C0 (TSC rate × ΔAPERF / ΔMPERF).
        pub busy_mhz: Option<f64>,
    }

    /// Derive interval clocks. Returns `None` when a counter went backwards
    /// (CPU hotplug, resume) or no time elapsed.
    pub(super) fn clock_stats(prev: &Counters, cur: &Counters) -> Option<ClockStats> {
        let secs = cur.at.checked_duration_since(prev.at)?.as_secs_f64();
        let d_tsc = cur.tsc.checked_sub(prev.tsc)?;
        let d_aperf = cur.aperf.checked_sub(prev.aperf)?;
        let d_mperf = cur.mperf.checked_sub(prev.mperf)?;
        if secs <= 0.0 || d_tsc == 0 {
            return None;
        }
        let tsc_mhz = d_tsc as f64 / secs / 1e6;
        Some(ClockStats {
            effective_mhz: d_aperf as f64 / secs / 1e6,
            busy_pct: (d_mperf as f64 / d_tsc as f64 * 100.0).min(100.0),
            busy_mhz: (d_mperf > 0).then(|| tsc_mhz * d_aperf as f64 / d_mperf as f64),
        })
    }

    // ── Source ──────────────────────────────────────────────────────────

    /// Hybrid core type of a logical CPU, used for labels.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum CoreKind {
        Performance,
        Efficient,
        Uniform,
    }

    pub(super) fn core_label(kind: CoreKind, cpu: u32) -> String {
        match kind {
            CoreKind::Performance => format!("P-Core {cpu}"),
            CoreKind::Efficient => format!("E-Core {cpu}"),
            CoreKind::Uniform => format!("Core {cpu}"),
        }
    }

    struct CpuEntry {
        cpu: u32,
        prefix: String,
        dev: MsrDevice,
        prev: Option<Counters>,
    }

    struct PackageEntry {
        id: u32,
        dev: MsrDevice,
        /// Previous `(TSC, package C2 residency)` sample.
        prev_c2: Option<(u64, u64)>,
    }

    pub struct IntelMsrSource {
        cpus: Vec<CpuEntry>,
        packages: Vec<PackageEntry>,
        features: ModelFeatures,
        aperfmperf: bool,
    }

    impl IntelMsrSource {
        pub fn discover() -> Self {
            let empty = Self {
                cpus: Vec::new(),
                packages: Vec::new(),
                features: ModelFeatures::default(),
                aperfmperf: false,
            };

            let cpuinfo = crate::platform::procfs::parse_cpuinfo();
            let Some(first) = cpuinfo.first() else {
                return empty;
            };
            if first.get("vendor_id").map(String::as_str) != Some("GenuineIntel") {
                return empty;
            }
            let family = first
                .get("cpu family")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let model = first.get("model").and_then(|s| s.parse().ok()).unwrap_or(0);
            let flags: Vec<&str> = first
                .get("flags")
                .map(|f| f.split_whitespace().collect())
                .unwrap_or_default();
            let aperfmperf = flags.contains(&"aperfmperf");

            let online = sysfs::read_string_optional(Path::new("/sys/devices/system/cpu/online"))
                .map(|s| sysfs::parse_cpulist(&s))
                .unwrap_or_default();
            let p_cores = read_cpulist("/sys/devices/cpu_core/cpus");
            let e_cores = read_cpulist("/sys/devices/cpu_atom/cpus");

            let mut cpus = Vec::new();
            let mut packages: Vec<PackageEntry> = Vec::new();
            for cpu in online {
                let Some(dev) = MsrDevice::open(cpu) else {
                    continue;
                };
                let kind = if p_cores.contains(&cpu) {
                    CoreKind::Performance
                } else if e_cores.contains(&cpu) {
                    CoreKind::Efficient
                } else {
                    CoreKind::Uniform
                };
                let pkg = sysfs::read_u32_optional(Path::new(&format!(
                    "/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
                )))
                .unwrap_or(0);
                if !packages.iter().any(|p| p.id == pkg)
                    && let Some(pkg_dev) = MsrDevice::open(cpu)
                {
                    let prev_c2 = read_pkg_c2(&pkg_dev);
                    packages.push(PackageEntry {
                        id: pkg,
                        dev: pkg_dev,
                        prev_c2,
                    });
                }
                let mut entry = CpuEntry {
                    cpu,
                    prefix: core_label(kind, cpu),
                    dev,
                    prev: None,
                };
                // Baseline now so the first poll already has a delta.
                if aperfmperf {
                    entry.prev = read_counters(&entry.dev);
                }
                cpus.push(entry);
            }

            if cpus.is_empty() {
                log::debug!("Intel MSR: /dev/cpu/*/msr not readable");
                return empty;
            }
            let features = model_features(family, model);
            log::info!(
                "Intel MSR: {} CPUs, {} packages, model {:#x} (core VID: {}, uncore VID: {})",
                cpus.len(),
                packages.len(),
                model,
                features.core_vid,
                features.uncore_vid
            );
            Self {
                cpus,
                packages,
                features,
                aperfmperf,
            }
        }

        pub fn poll(&mut self) -> Vec<(SensorId, SensorReading)> {
            let mut readings = Vec::new();
            for entry in &mut self.cpus {
                poll_cpu(entry, self.features, self.aperfmperf, &mut readings);
            }
            for pkg in &mut self.packages {
                poll_package(pkg, self.features, &mut readings);
            }
            readings
        }
    }

    fn read_cpulist(path: &str) -> Vec<u32> {
        sysfs::read_string_optional(Path::new(path))
            .map(|s| sysfs::parse_cpulist(&s))
            .unwrap_or_default()
    }

    fn read_pkg_c2(dev: &MsrDevice) -> Option<(u64, u64)> {
        Some((
            dev.read(msr::IA32_TIME_STAMP_COUNTER)?,
            dev.read(msr::MSR_PKG_C2_RESIDENCY)?,
        ))
    }

    fn read_counters(dev: &MsrDevice) -> Option<Counters> {
        let at = Instant::now();
        Some(Counters {
            tsc: dev.read(msr::IA32_TIME_STAMP_COUNTER)?,
            aperf: dev.read(msr::IA32_APERF)?,
            mperf: dev.read(msr::IA32_MPERF)?,
            at,
        })
    }

    fn poll_cpu(
        entry: &mut CpuEntry,
        features: ModelFeatures,
        aperfmperf: bool,
        out: &mut Vec<(SensorId, SensorReading)>,
    ) {
        let cpu = entry.cpu;
        let prefix = &entry.prefix;

        if aperfmperf && let Some(cur) = read_counters(&entry.dev) {
            if let Some(stats) = entry.prev.as_ref().and_then(|p| clock_stats(p, &cur)) {
                out.push(reading(
                    "eff_clock",
                    cpu,
                    format!("{prefix} Effective Clock"),
                    stats.effective_mhz,
                    SensorUnit::Mhz,
                    SensorCategory::Frequency,
                ));
                out.push(reading(
                    "busy",
                    cpu,
                    format!("{prefix} Busy"),
                    stats.busy_pct,
                    SensorUnit::Percent,
                    SensorCategory::Utilization,
                ));
                if let Some(mhz) = stats.busy_mhz {
                    out.push(reading(
                        "busy_clock",
                        cpu,
                        format!("{prefix} Busy Clock"),
                        mhz,
                        SensorUnit::Mhz,
                        SensorCategory::Frequency,
                    ));
                }
            }
            entry.prev = Some(cur);
        }

        if let Some(perf_status) = entry.dev.read(msr::IA32_PERF_STATUS) {
            if let Some(ratio) = core_ratio(perf_status) {
                out.push(reading(
                    "ratio",
                    cpu,
                    format!("{prefix} Ratio"),
                    ratio as f64,
                    SensorUnit::Unitless,
                    SensorCategory::Other,
                ));
            }
            if features.core_vid
                && let Some(volts) = core_vid_volts(perf_status)
            {
                out.push(reading(
                    "vid",
                    cpu,
                    format!("{prefix} VID"),
                    volts,
                    SensorUnit::Volts,
                    SensorCategory::Voltage,
                ));
            }
        }

        if let Some(status) = entry.dev.read(msr::IA32_THERM_STATUS) {
            if let Some(margin) = distance_to_tjmax(status) {
                out.push(reading(
                    "tjmax_distance",
                    cpu,
                    format!("{prefix} Distance to TjMAX"),
                    margin as f64,
                    SensorUnit::Celsius,
                    SensorCategory::Other,
                ));
            }
            out.push(reading(
                "therm_status",
                cpu,
                format!("{prefix} Thermal Status"),
                therm_flags(status) as f64,
                SensorUnit::Bitmask,
                SensorCategory::Other,
            ));
        }
    }

    fn poll_package(
        pkg: &mut PackageEntry,
        features: ModelFeatures,
        out: &mut Vec<(SensorId, SensorReading)>,
    ) {
        let chip = format!("package{}", pkg.id);
        let c2_now = read_pkg_c2(&pkg.dev);
        let c2_pct = pkg
            .prev_c2
            .zip(c2_now)
            .and_then(|(p, c)| residency_pct(p, c));
        pkg.prev_c2 = c2_now;
        let mut push = |sensor: &str, label: &str, value: f64, unit, category| {
            out.push((
                SensorId {
                    source: "cpu".into(),
                    chip: chip.clone(),
                    sensor: sensor.into(),
                },
                SensorReading::new(label.into(), value, unit, category),
            ));
        };

        if features.uncore_vid
            && let Some(status) = pkg.dev.read(msr::MSR_UNCORE_PERF_STATUS)
        {
            if let Some(volts) = uncore_vid_volts(status) {
                push(
                    "uncore_vid",
                    "Uncore VID",
                    volts,
                    SensorUnit::Volts,
                    SensorCategory::Voltage,
                );
            }
            if let Some(ratio) = ring_ratio(status) {
                push(
                    "ring_ratio",
                    "Ring Ratio",
                    ratio as f64,
                    SensorUnit::Unitless,
                    SensorCategory::Other,
                );
            }
        }

        if let Some((min, max)) = pkg
            .dev
            .read(msr::MSR_RING_RATIO_LIMIT)
            .and_then(ring_ratio_limits)
        {
            push(
                "ring_ratio_min",
                "Ring Ratio Min",
                min as f64,
                SensorUnit::Unitless,
                SensorCategory::Other,
            );
            push(
                "ring_ratio_max",
                "Ring Ratio Max",
                max as f64,
                SensorUnit::Unitless,
                SensorCategory::Other,
            );
        }

        if let Some(pct) = c2_pct {
            push(
                "c2_residency",
                "Package C2 Residency",
                pct,
                SensorUnit::Percent,
                SensorCategory::Other,
            );
        }

        if let Some(target) = pkg.dev.read(msr::MSR_TEMPERATURE_TARGET)
            && let Some(tj) = tjmax(target)
        {
            push(
                "tjmax",
                "TjMAX",
                tj as f64,
                SensorUnit::Celsius,
                SensorCategory::Other,
            );
            push(
                "tcc_offset",
                "TCC Offset",
                tcc_offset(target) as f64,
                SensorUnit::Celsius,
                SensorCategory::Other,
            );
        }

        if let Some(status) = pkg.dev.read(msr::IA32_PACKAGE_THERM_STATUS) {
            if let Some(margin) = distance_to_tjmax(status) {
                push(
                    "tjmax_distance",
                    "Package Distance to TjMAX",
                    margin as f64,
                    SensorUnit::Celsius,
                    SensorCategory::Other,
                );
            }
            push(
                "therm_status",
                "Package Thermal Status",
                therm_flags(status) as f64,
                SensorUnit::Bitmask,
                SensorCategory::Other,
            );
        }

        if let Some(amps) = pkg
            .dev
            .read(msr::MSR_VR_CURRENT_CONFIG)
            .and_then(vr_current_limit_amps)
        {
            push(
                "vr_current_limit",
                "VR Current Limit",
                amps,
                SensorUnit::Amps,
                SensorCategory::Current,
            );
        }

        for (addr, sensor, label) in [
            (
                msr::MSR_CORE_PERF_LIMIT_REASONS,
                "ia_limit_reasons",
                "IA Limit Reasons",
            ),
            (
                msr::MSR_GRAPHICS_PERF_LIMIT_REASONS,
                "gt_limit_reasons",
                "GT Limit Reasons",
            ),
            (
                msr::MSR_RING_PERF_LIMIT_REASONS,
                "ring_limit_reasons",
                "Ring Limit Reasons",
            ),
        ] {
            if let Some(raw) = pkg.dev.read(addr) {
                // Bits 15:0 are live status, 31:16 sticky logs; shown raw.
                push(
                    sensor,
                    label,
                    bits(raw, 31, 0) as f64,
                    SensorUnit::Bitmask,
                    SensorCategory::Other,
                );
            }
        }
    }

    fn reading(
        chip: &str,
        cpu: u32,
        label: String,
        value: f64,
        unit: SensorUnit,
        category: SensorCategory,
    ) -> (SensorId, SensorReading) {
        (
            SensorId {
                source: "cpu".into(),
                chip: chip.into(),
                sensor: format!("cpu{cpu}"),
            },
            SensorReading::new(label, value, unit, category),
        )
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod inner {
    use crate::model::sensor::{SensorId, SensorReading};

    pub struct IntelMsrSource;

    impl IntelMsrSource {
        pub fn discover() -> Self {
            Self
        }

        pub fn poll(&mut self) -> Vec<(SensorId, SensorReading)> {
            Vec::new()
        }
    }
}

pub use inner::IntelMsrSource;

impl crate::sensors::SensorSource for IntelMsrSource {
    fn name(&self) -> &str {
        "intel_msr"
    }

    fn poll(&mut self) -> Vec<(SensorId, SensorReading)> {
        IntelMsrSource::poll(self)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::inner::*;
    use std::time::{Duration, Instant};

    // Raw words captured on a Core Ultra 9 285K (Arrow Lake-S, model 0xC6).

    #[test]
    fn test_core_vid_and_ratio_pcore_idle() {
        let raw = 0x0000_1605_0000_0800;
        assert!((core_vid_volts(raw).unwrap() - 0.688).abs() < 0.0005);
        assert_eq!(core_ratio(raw), Some(8));
    }

    #[test]
    fn test_core_vid_and_ratio_ecore() {
        let raw = 0x0000_23EC_0000_3000;
        assert!((core_vid_volts(raw).unwrap() - 1.1226).abs() < 0.0005);
        assert_eq!(core_ratio(raw), Some(48));
    }

    #[test]
    fn test_zero_vid_is_absent_not_zero_volts() {
        assert_eq!(core_vid_volts(0x0000_0000_0000_2A00), None);
        assert_eq!(core_ratio(0), None);
    }

    #[test]
    fn test_uncore_vid_and_ring_ratio() {
        let raw = 0x26EB_0028;
        assert!((uncore_vid_volts(raw).unwrap() - 1.2162).abs() < 0.0005);
        assert_eq!(ring_ratio(raw), Some(40));
    }

    #[test]
    fn test_tjmax_and_tcc_offset() {
        assert_eq!(tjmax(0x8069_1400), Some(105));
        assert_eq!(tcc_offset(0x8069_1400), 0);
        assert_eq!(tcc_offset(0x8A69_1400), 10);
        assert_eq!(tjmax(0), None);
    }

    #[test]
    fn test_distance_to_tjmax_requires_valid_bit() {
        // Core 0: readout 0x47 = 71 below TjMAX 105 -> 34 °C
        assert_eq!(distance_to_tjmax(0x8847_2000), Some(71));
        assert_eq!(distance_to_tjmax(0x0847_2000), None);
        assert_eq!(therm_flags(0x8847_2003), 0x2003);
    }

    #[test]
    fn test_ring_ratio_limits() {
        // 0x0828: min 8x, max 40x
        assert_eq!(ring_ratio_limits(0x0828), Some((8, 40)));
        assert_eq!(ring_ratio_limits(0), None);
    }

    #[test]
    fn test_residency_pct() {
        assert_eq!(residency_pct((0, 0), (1000, 250)), Some(25.0));
        assert_eq!(residency_pct((1000, 0), (1000, 10)), None);
        assert_eq!(residency_pct((1000, 50), (2000, 10)), None);
    }

    #[test]
    fn test_vr_current_limit_is_amps() {
        assert_eq!(vr_current_limit_amps(0xD48), Some(425.0));
        assert_eq!(vr_current_limit_amps(0), None);
    }

    #[test]
    fn test_model_features_gate() {
        let arl = model_features(6, 0xC6);
        assert!(arl.core_vid && arl.uncore_vid);
        let rpl = model_features(6, 0xB7);
        assert!(rpl.core_vid && !rpl.uncore_vid);
        // Arrow Lake-H/U and unknown models stay off until verified
        assert_eq!(model_features(6, 0xC5), ModelFeatures::default());
        assert_eq!(model_features(0x19, 0xC6), ModelFeatures::default());
    }

    #[test]
    fn test_clock_stats() {
        let t0 = Instant::now();
        let prev = Counters {
            tsc: 0,
            aperf: 0,
            mperf: 0,
            at: t0,
        };
        // 1 s at a 3700 MHz TSC, 25% in C0 at 5000 MHz
        let cur = Counters {
            tsc: 3_700_000_000,
            aperf: 1_250_000_000,
            mperf: 925_000_000,
            at: t0 + Duration::from_secs(1),
        };
        let s = clock_stats(&prev, &cur).unwrap();
        assert!((s.effective_mhz - 1250.0).abs() < 0.01);
        assert!((s.busy_pct - 25.0).abs() < 0.01);
        assert!((s.busy_mhz.unwrap() - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_clock_stats_rejects_backwards_counters() {
        let t0 = Instant::now();
        let prev = Counters {
            tsc: 100,
            aperf: 100,
            mperf: 100,
            at: t0,
        };
        let cur = Counters {
            tsc: 50,
            aperf: 200,
            mperf: 200,
            at: t0 + Duration::from_secs(1),
        };
        assert!(clock_stats(&prev, &cur).is_none());
    }

    #[test]
    fn test_clock_stats_idle_has_no_busy_clock() {
        let t0 = Instant::now();
        let prev = Counters {
            tsc: 0,
            aperf: 0,
            mperf: 0,
            at: t0,
        };
        let cur = Counters {
            tsc: 1_000_000,
            aperf: 0,
            mperf: 0,
            at: t0 + Duration::from_millis(10),
        };
        let s = clock_stats(&prev, &cur).unwrap();
        assert_eq!(s.busy_pct, 0.0);
        assert_eq!(s.busy_mhz, None);
    }

    #[test]
    fn test_core_labels() {
        assert_eq!(core_label(CoreKind::Performance, 3), "P-Core 3");
        assert_eq!(core_label(CoreKind::Efficient, 8), "E-Core 8");
        assert_eq!(core_label(CoreKind::Uniform, 0), "Core 0");
    }
}
