//! Intel uncore clocks and DRAM bandwidth from uncore perf counters.
//!
//! - **CNCU / SNCU clocks** (Meteor Lake and later): the fixed `clockticks`
//!   counters of the `uncore_cncu` and `uncore_sncu` PMUs, as an
//!   interval-average MHz. On Arrow Lake-S, CNCU is the ring/LLC clock: it
//!   holds exactly 4000 MHz under all-core load with the ring pinned at 40x,
//!   unlike the ring *ratio* in `MSR_UNCORE_PERF_STATUS` which is a request.
//!   SNCU is a ratio-locked SoC fabric clock, but which one is unresolved:
//!   it measured 3200 MHz with the BIOS D2D and NGU ratios both at 32x, and
//!   those are separate knobs for different parts of the chip. Telling them
//!   apart needs a run with the two ratios set differently, so it is
//!   labelled by its PMU name only.
//! - **DRAM bandwidth** (Arrow Lake-S only): the `uncore_imc_N` data events
//!   `0x3A` (read), `0x3B` (write) and `0x3C` (total), 32 bytes per count,
//!   summed over both memory controllers.
//!
//! Counters run continuously; each poll reports the delta since the last
//! one. A counter the kernel had to multiplex for more than 5% of the
//! interval is skipped for that poll rather than extrapolated.

use crate::model::sensor::{SensorCategory, SensorId, SensorReading, SensorUnit};
use crate::platform::perf::{self, PerfCounter, PerfSample};

/// Minimum running/enabled ratio for a counter delta to be published.
const MIN_RUNNING: f64 = 0.95;

/// Bytes transferred per IMC data event count (Arrow Lake-S).
const IMC_BYTES_PER_COUNT: f64 = 32.0;

struct Counter {
    counter: PerfCounter,
    prev: Option<PerfSample>,
}

impl Counter {
    fn open(pmu: &str, event: &str) -> Option<Self> {
        let mut counter = PerfCounter::open(pmu, event)?;
        let prev = counter.read();
        Some(Self { counter, prev })
    }

    /// Events per second since the previous sample, then advance the
    /// baseline.
    fn rate(&mut self) -> Option<f64> {
        let cur = self.counter.read()?;
        let rate = self
            .prev
            .as_ref()
            .and_then(|prev| perf::rate(prev, &cur, MIN_RUNNING));
        self.prev = Some(cur);
        rate
    }
}

/// A measured quantity built from one or more counters that are summed.
struct Metric {
    id: SensorId,
    label: &'static str,
    unit: SensorUnit,
    category: SensorCategory,
    /// Converts summed events per second into the published unit.
    scale: f64,
    counters: Vec<Counter>,
}

pub struct IntelUncoreSource {
    metrics: Vec<Metric>,
}

/// Uncore clock PMUs and their labels.
const CLOCK_PMUS: &[(&str, &str, &str)] = &[
    ("uncore_cncu", "ring_clock", "Ring/LLC Clock (CNCU)"),
    ("uncore_sncu", "sncu_clock", "SNCU Clock"),
];

/// Arrow Lake-S IMC data events.
const IMC_EVENTS: &[(&str, &str, &str)] = &[
    ("event=0x3a", "dram_read", "DRAM Read"),
    ("event=0x3b", "dram_write", "DRAM Write"),
    ("event=0x3c", "dram_total", "DRAM Total"),
];

fn is_arrow_lake_s() -> bool {
    let cpuinfo = crate::platform::procfs::parse_cpuinfo();
    let Some(first) = cpuinfo.first() else {
        return false;
    };
    first.get("vendor_id").map(String::as_str) == Some("GenuineIntel")
        && first.get("cpu family").map(String::as_str) == Some("6")
        && first.get("model").map(String::as_str) == Some("198")
}

fn id(chip: &str, sensor: &str) -> SensorId {
    SensorId {
        source: "uncore".into(),
        chip: chip.into(),
        sensor: sensor.into(),
    }
}

impl IntelUncoreSource {
    pub fn discover() -> Self {
        let mut metrics = Vec::new();

        for &(pmu, sensor, label) in CLOCK_PMUS {
            let Some(event) = perf::event_alias(pmu, "clockticks") else {
                continue;
            };
            if let Some(counter) = Counter::open(pmu, &event) {
                metrics.push(Metric {
                    id: id("clocks", sensor),
                    label,
                    unit: SensorUnit::Mhz,
                    category: SensorCategory::Frequency,
                    scale: 1e-6,
                    counters: vec![counter],
                });
            }
        }

        let imcs = perf::pmus_with_prefix("uncore_imc_")
            .into_iter()
            .filter(|n| n["uncore_imc_".len()..].chars().all(|c| c.is_ascii_digit()))
            .collect::<Vec<_>>();
        if !imcs.is_empty() && is_arrow_lake_s() {
            for &(event, sensor, label) in IMC_EVENTS {
                let counters: Vec<Counter> = imcs
                    .iter()
                    .filter_map(|pmu| Counter::open(pmu, event))
                    .collect();
                // Publish only when every controller is counted.
                if counters.len() == imcs.len() {
                    metrics.push(Metric {
                        id: id("imc", sensor),
                        label,
                        unit: SensorUnit::MegabytesPerSec,
                        category: SensorCategory::Throughput,
                        scale: IMC_BYTES_PER_COUNT / 1e6,
                        counters,
                    });
                }
            }
        }

        if !metrics.is_empty() {
            log::info!("Intel uncore: {} perf metrics", metrics.len());
        }
        Self { metrics }
    }

    pub fn poll(&mut self) -> Vec<(SensorId, SensorReading)> {
        let mut readings = Vec::new();
        for metric in &mut self.metrics {
            let rates: Vec<Option<f64>> = metric.counters.iter_mut().map(|c| c.rate()).collect();
            // A partial sum would under-report, so skip the poll instead.
            if rates.iter().any(Option::is_none) {
                continue;
            }
            let total: f64 = rates.into_iter().flatten().sum();
            readings.push((
                metric.id.clone(),
                SensorReading::new(
                    metric.label.into(),
                    total * metric.scale,
                    metric.unit,
                    metric.category,
                ),
            ));
        }
        readings
    }
}

impl crate::sensors::SensorSource for IntelUncoreSource {
    fn name(&self) -> &str {
        "intel_uncore"
    }

    fn poll(&mut self) -> Vec<(SensorId, SensorReading)> {
        IntelUncoreSource::poll(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imc_scale_matches_recorded_counts() {
        // Recorded on a 285K: 17,365,742 read events on one IMC in 1.0009 s
        let mb_s: f64 = 17_365_742.0 / 1.000_909 * IMC_BYTES_PER_COUNT / 1e6;
        assert!((mb_s - 555.2).abs() < 0.1);
    }

    #[test]
    fn test_clock_scale() {
        // Recorded: CNCU 3,556,583,706 ticks in 1.0009 s -> ~3553 MHz
        let mhz: f64 = 3_556_583_706.0 / 1.000_902_533 * 1e-6;
        assert!((mhz - 3553.4).abs() < 0.1);
    }
}
