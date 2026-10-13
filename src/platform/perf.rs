//! Counting-mode `perf_event_open` for uncore PMUs.
//!
//! Opens system-wide counters on PMUs listed under
//! `/sys/bus/event_source/devices/`. The kernel programs, multiplexes and
//! extends the hardware counters; this module only reads counts. Event
//! strings use the sysfs syntax (`event=0x3a`, `event=0xff,umask=0x20`) and
//! are encoded through each PMU's `format/` descriptions.
//!
//! System-wide uncore events need `CAP_PERFMON` (or root) whenever
//! `kernel.perf_event_paranoid` is above 0.

use std::fs::File;
use std::io::Read;
use std::os::unix::io::FromRawFd;
use std::path::{Path, PathBuf};

const PMU_ROOT: &str = "/sys/bus/event_source/devices";

const PERF_FORMAT_TOTAL_TIME_ENABLED: u64 = 1 << 0;
const PERF_FORMAT_TOTAL_TIME_RUNNING: u64 = 1 << 1;
const PERF_FLAG_FD_CLOEXEC: libc::c_ulong = 1 << 3;

/// `struct perf_event_attr` up to `config1` (`PERF_ATTR_SIZE_VER0`, 64
/// bytes). The kernel zero-extends the fields this version predates.
#[repr(C)]
#[derive(Default)]
struct PerfEventAttr {
    kind: u32,
    size: u32,
    config: u64,
    sample_period: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup_events: u32,
    bp_type: u32,
    config1: u64,
}

/// One counter sample: raw count plus the time the event was enabled and
/// actually counting (they differ when the kernel multiplexes counters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfSample {
    pub value: u64,
    pub time_enabled: u64,
    pub time_running: u64,
}

/// A counting perf event; closed on drop.
pub struct PerfCounter {
    file: File,
}

impl PerfCounter {
    /// Open `event` (sysfs syntax) system-wide on `pmu`, on the first CPU of
    /// the PMU's `cpumask`. Returns `None` if the PMU or a format term is
    /// unknown, or the kernel refuses the event (permissions, no counter).
    pub fn open(pmu: &str, event: &str) -> Option<Self> {
        let dir = pmu_dir(pmu);
        let kind = read_trimmed(&dir.join("type"))?.parse::<u32>().ok()?;
        let config = encode_config(&dir, event)?;
        let cpu = read_trimmed(&dir.join("cpumask"))
            .and_then(|mask| {
                crate::platform::sysfs::parse_cpulist(&mask)
                    .first()
                    .copied()
            })
            .unwrap_or(0);

        let attr = PerfEventAttr {
            kind,
            size: std::mem::size_of::<PerfEventAttr>() as u32,
            config,
            read_format: PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
            ..Default::default()
        };
        // SAFETY: `attr` is a valid, fully initialised perf_event_attr of the
        // size it declares; the kernel only reads it.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &attr as *const PerfEventAttr,
                -1 as libc::pid_t,
                cpu as libc::c_int,
                -1 as libc::c_int,
                PERF_FLAG_FD_CLOEXEC,
            )
        };
        if fd < 0 {
            log::debug!(
                "perf_event_open({pmu}/{event}/) failed: {}",
                std::io::Error::last_os_error()
            );
            return None;
        }
        // SAFETY: the syscall returned a new file descriptor that we own.
        let file = unsafe { File::from_raw_fd(fd as libc::c_int) };
        Some(Self { file })
    }

    /// Read the current count and enabled/running times.
    pub fn read(&mut self) -> Option<PerfSample> {
        let mut buf = [0u8; 24];
        self.file.read_exact(&mut buf).ok()?;
        let word = |i: usize| u64::from_ne_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
        Some(PerfSample {
            value: word(0),
            time_enabled: word(1),
            time_running: word(2),
        })
    }
}

/// Whether a PMU with this name is registered.
pub fn pmu_exists(pmu: &str) -> bool {
    pmu_dir(pmu).join("type").exists()
}

/// Names of registered PMUs whose name starts with `prefix`, sorted.
pub fn pmus_with_prefix(prefix: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(PMU_ROOT)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix))
        .collect();
    names.sort_by(|a, b| crate::model::sensor::natural_cmp_str(a, b));
    names
}

/// The sysfs event string of a named event alias (`events/<name>`).
pub fn event_alias(pmu: &str, name: &str) -> Option<String> {
    read_trimmed(&pmu_dir(pmu).join("events").join(name))
}

/// Events per second between two samples, timed by the kernel's enabled
/// time. Returns `None` when the counter went backwards or ran for less than
/// `min_running` of the enabled time (multiplexed); multiplexed counts are
/// not extrapolated.
pub fn rate(prev: &PerfSample, cur: &PerfSample, min_running: f64) -> Option<f64> {
    let d_value = cur.value.checked_sub(prev.value)?;
    let d_enabled = cur.time_enabled.checked_sub(prev.time_enabled)?;
    let d_running = cur.time_running.checked_sub(prev.time_running)?;
    if d_enabled == 0 || (d_running as f64) < d_enabled as f64 * min_running {
        return None;
    }
    Some(d_value as f64 / (d_enabled as f64 / 1e9))
}

fn pmu_dir(pmu: &str) -> PathBuf {
    Path::new(PMU_ROOT).join(pmu)
}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Encode a sysfs event string into `attr.config` using the PMU's formats.
fn encode_config(pmu_dir: &Path, event: &str) -> Option<u64> {
    let mut config = 0u64;
    for term in event.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (name, value) = term.split_once('=').unwrap_or((term, "1"));
        let value = parse_value(value)?;
        let format = read_trimmed(&pmu_dir.join("format").join(name))?;
        let (lo, hi) = parse_format(&format)?;
        config |= encode_term(value, lo, hi);
    }
    Some(config)
}

fn parse_value(s: &str) -> Option<u64> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
}

/// Parse a single-range `config` format such as `config:8-15` or
/// `config:18`. Formats on `config1`/`config2` or split ranges are not
/// supported and return `None`.
pub(crate) fn parse_format(format: &str) -> Option<(u32, u32)> {
    let range = format.strip_prefix("config:")?;
    if range.contains(',') {
        return None;
    }
    match range.split_once('-') {
        Some((lo, hi)) => Some((lo.parse().ok()?, hi.parse().ok()?)),
        None => {
            let bit = range.parse().ok()?;
            Some((bit, bit))
        }
    }
}

pub(crate) fn encode_term(value: u64, lo: u32, hi: u32) -> u64 {
    let width = hi - lo + 1;
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    (value & mask) << lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attr_is_ver0_size() {
        assert_eq!(std::mem::size_of::<PerfEventAttr>(), 64);
    }

    #[test]
    fn test_parse_format() {
        assert_eq!(parse_format("config:0-7"), Some((0, 7)));
        assert_eq!(parse_format("config:8-15"), Some((8, 15)));
        assert_eq!(parse_format("config:18"), Some((18, 18)));
        assert_eq!(parse_format("config1:0-7"), None);
        assert_eq!(parse_format("config:0-7,32-35"), None);
    }

    #[test]
    fn test_encode_terms() {
        // uncore_imc_free_running data_read: event=0xff,umask=0x20
        let cfg = encode_term(0xff, 0, 7) | encode_term(0x20, 8, 15);
        assert_eq!(cfg, 0x20ff);
        // value wider than the field is truncated, never spills over
        assert_eq!(encode_term(0x1ff, 0, 7), 0xff);
    }

    #[test]
    fn test_rate_rejects_multiplexed_and_backwards() {
        let a = PerfSample {
            value: 100,
            time_enabled: 1_000_000_000,
            time_running: 1_000_000_000,
        };
        // 500 events over 0.5 s of enabled time
        let full = PerfSample {
            value: 600,
            time_enabled: 1_500_000_000,
            time_running: 1_500_000_000,
        };
        assert_eq!(rate(&a, &full, 0.95), Some(1000.0));
        let multiplexed = PerfSample {
            value: 600,
            time_enabled: 1_500_000_000,
            time_running: 1_200_000_000,
        };
        assert_eq!(rate(&a, &multiplexed, 0.95), None);
        assert_eq!(rate(&full, &a, 0.95), None);
    }

    #[test]
    fn test_parse_value() {
        assert_eq!(parse_value("0x3a"), Some(0x3a));
        assert_eq!(parse_value("58"), Some(58));
        assert_eq!(parse_value("zz"), None);
    }
}
