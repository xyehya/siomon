//! Dedicated CPU detail view for the TUI.
//!
//! A per-core table (effective and busy clock, C0 residency, ratio,
//! requested VID, DTS temperature, distance to TjMAX, thermal flags) with
//! P-core/E-core min/avg/max rows, next to package panels for clocks,
//! voltages, power, thermals, memory and raw limit reasons. Built from the
//! `intel_msr`, `intel_uncore`, `intel_mchbar`, `intel_gpu`, RAPL and hwmon
//! readings plus the trained memory-controller info.
//!
//! VIDs are shown as requests and board-sensed rails as measurements, in
//! separate groups, because they are different quantities.

use std::collections::HashMap;
use std::io::{self, Stdout};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};

use crate::model::memory::MemoryInfo;
use crate::model::sensor::{SensorCategory, SensorId, SensorReading};

use super::SystemSummary;
use super::theme::TuiTheme;

/// Hybrid core type, taken from the `P-Core`/`E-Core` label prefix the
/// MSR source emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CoreKind {
    Performance,
    Efficient,
    Uniform,
}

impl CoreKind {
    fn from_label(label: &str) -> Self {
        if label.starts_with("P-Core") {
            Self::Performance
        } else if label.starts_with("E-Core") {
            Self::Efficient
        } else {
            Self::Uniform
        }
    }

    fn short(self) -> &'static str {
        match self {
            Self::Performance => "P",
            Self::Efficient => "E",
            Self::Uniform => "",
        }
    }
}

/// One logical CPU's row.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct CoreRow {
    pub cpu: u32,
    pub kind: Option<CoreKind>,
    pub eff_mhz: Option<f64>,
    pub busy_mhz: Option<f64>,
    pub busy_pct: Option<f64>,
    pub ratio: Option<f64>,
    pub vid: Option<f64>,
    pub temp: Option<f64>,
    pub margin: Option<f64>,
    pub flags: Option<u64>,
}

/// Readings keyed by `source/chip/sensor` for direct lookup.
pub(crate) struct Index<'a> {
    map: HashMap<String, &'a SensorReading>,
}

impl<'a> Index<'a> {
    pub(crate) fn new(snapshot: &'a [(SensorId, SensorReading)]) -> Self {
        Self {
            map: snapshot.iter().map(|(id, r)| (id.to_string(), r)).collect(),
        }
    }

    fn get(&self, key: &str) -> Option<&'a SensorReading> {
        self.map.get(key).copied()
    }

    fn value(&self, key: &str) -> Option<f64> {
        self.get(key).map(|r| r.current)
    }
}

/// Per-CPU chips emitted by the MSR source.
const CORE_CHIPS: &[&str] = &[
    "eff_clock",
    "busy_clock",
    "busy",
    "ratio",
    "vid",
    "tjmax_distance",
    "therm_status",
];

/// Build per-core rows from the snapshot, sorted by CPU number.
pub(crate) fn core_rows(snapshot: &[(SensorId, SensorReading)]) -> Vec<CoreRow> {
    let idx = Index::new(snapshot);
    let tjmax = idx.value("cpu/package0/tjmax");
    let mut cpus: Vec<(u32, Option<CoreKind>)> = Vec::new();
    for (id, r) in snapshot {
        if id.source != "cpu" || !CORE_CHIPS.contains(&id.chip.as_str()) {
            continue;
        }
        let Some(cpu) = id.sensor.strip_prefix("cpu").and_then(|n| n.parse().ok()) else {
            continue;
        };
        if !cpus.iter().any(|(c, _)| *c == cpu) {
            cpus.push((cpu, Some(CoreKind::from_label(&r.label))));
        }
    }
    cpus.sort_unstable();

    cpus.into_iter()
        .map(|(cpu, kind)| {
            let v = |chip: &str| idx.value(&format!("cpu/{chip}/cpu{cpu}"));
            let margin = v("tjmax_distance");
            CoreRow {
                cpu,
                kind,
                eff_mhz: v("eff_clock"),
                busy_mhz: v("busy_clock"),
                busy_pct: v("busy"),
                ratio: v("ratio"),
                vid: v("vid"),
                temp: tjmax.zip(margin).map(|(tj, m)| tj - m),
                margin,
                flags: v("therm_status").map(|f| f as u64),
            }
        })
        .collect()
}

/// Min / average / max of the present values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Stats {
    pub min: f64,
    pub avg: f64,
    pub max: f64,
}

pub(crate) fn stats(values: impl Iterator<Item = Option<f64>>) -> Option<Stats> {
    let vals: Vec<f64> = values.flatten().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return None;
    }
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let avg = vals.iter().sum::<f64>() / vals.len() as f64;
    Some(Stats { min, avg, max })
}

/// Thermal status bits (IA32_THERM_STATUS / IA32_PACKAGE_THERM_STATUS):
/// even bits are live status, the next odd bit is its sticky log.
const THERM_FLAGS: &[(u32, &str)] = &[
    (0, "TH"),
    (2, "PH"),
    (4, "CR"),
    (6, "T1"),
    (8, "T2"),
    (10, "PL"),
    (12, "CL"),
    (14, "XD"),
];

/// Short codes for a thermal status word: uppercase for live status,
/// lowercase for a sticky log bit (set since boot, never cleared here).
pub(crate) fn therm_flag_codes(word: u64) -> String {
    let mut codes = Vec::new();
    for &(bit, code) in THERM_FLAGS {
        if word >> bit & 1 == 1 {
            codes.push(code.to_string());
        }
        if word >> (bit + 1) & 1 == 1 {
            codes.push(code.to_lowercase());
        }
    }
    codes.join(" ")
}

/// Set bit numbers within `lo..=hi`.
pub(crate) fn set_bits(word: u64, lo: u32, hi: u32) -> Vec<u32> {
    (lo..=hi).filter(|&b| word >> b & 1 == 1).collect()
}

fn bit_list(bits: &[u32]) -> String {
    if bits.is_empty() {
        "-".into()
    } else {
        bits.iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn fmt(v: Option<f64>, prec: usize) -> String {
    v.map(|v| format!("{v:.prec$}"))
        .unwrap_or_else(|| "-".into())
}

fn temp_style(theme: &TuiTheme, t: Option<f64>) -> Style {
    match t {
        Some(t) if t > 80.0 => Style::default().fg(theme.crit).add_modifier(Modifier::BOLD),
        Some(t) if t >= 60.0 => Style::default().fg(theme.warn),
        Some(_) => Style::default().fg(theme.good),
        None => theme.muted_style(),
    }
}

/// A labelled value line for the side panels.
fn kv<'a>(theme: &TuiTheme, label: &str, value: String, style: Style) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{label:<20}"), theme.label_style()),
        Span::styled(value, style),
    ])
}

fn heading<'a>(theme: &TuiTheme, text: &str) -> Line<'a> {
    Line::from(Span::styled(
        text.to_string(),
        theme.muted_style().add_modifier(Modifier::BOLD),
    ))
}

struct Panel<'a> {
    title: &'static str,
    lines: Vec<Line<'a>>,
}

/// Explain missing uncore perf readings when the PMU exists: either no
/// permission, or another perf session (e.g. `perf stat`, turbostat) holds
/// the counters, in which case readings are skipped rather than estimated.
fn uncore_hint<'a>(idx: &Index, prefix: &str, pmu: &str, theme: &TuiTheme) -> Option<Line<'a>> {
    let present = idx.map.keys().any(|k| k.starts_with(prefix));
    (!present && crate::platform::perf::pmu_exists(pmu)).then(|| {
        Line::from(Span::styled(
            "uncore counters unavailable: needs root, or in use by another perf session",
            theme.muted_style(),
        ))
    })
}

fn clocks_panel<'a>(idx: &Index, rows: &[CoreRow], theme: &TuiTheme) -> Panel<'a> {
    let mut lines = Vec::new();
    let freq = theme.info_style();
    for (kind, name) in [
        (CoreKind::Performance, "P-core eff (avg)"),
        (CoreKind::Efficient, "E-core eff (avg)"),
    ] {
        if let Some(s) = stats(
            rows.iter()
                .filter(|r| r.kind == Some(kind))
                .map(|r| r.eff_mhz),
        ) {
            lines.push(kv(theme, name, format!("{:.0} MHz", s.avg), freq));
        }
    }
    if let Some(r) = idx.value("cpu/package0/ring_ratio") {
        let limits = idx
            .value("cpu/package0/ring_ratio_min")
            .zip(idx.value("cpu/package0/ring_ratio_max"))
            .map(|(lo, hi)| format!(" ({lo:.0}-{hi:.0}x)"))
            .unwrap_or_default();
        lines.push(kv(
            theme,
            "Ring ratio (req)",
            format!("{r:.0}x{limits}"),
            freq,
        ));
    }
    for (key, label) in [
        ("uncore/clocks/ring_clock", "Ring/LLC (CNCU)"),
        ("uncore/clocks/sncu_clock", "SNCU"),
        ("mchbar/arl/psf0_clock", "PSF0"),
    ] {
        if let Some(v) = idx.value(key) {
            lines.push(kv(theme, label, format!("{v:.0} MHz"), freq));
        }
    }
    lines.extend(uncore_hint(idx, "uncore/clocks/", "uncore_cncu", theme));
    for (key, label) in [
        ("mchbar/arl/bclk", "BCLK"),
        ("mchbar/arl/soc_bclk", "SoC BCLK"),
    ] {
        if let Some(v) = idx.value(key) {
            lines.push(kv(theme, label, format!("{v:.3} MHz"), freq));
        }
    }
    let mut gts: Vec<&str> = idx
        .map
        .keys()
        .filter_map(|k| k.strip_prefix("intel_gpu/")?.strip_suffix("/act_freq"))
        .collect();
    gts.sort_unstable();
    for gt in gts {
        let Some(clock) = idx.get(&format!("intel_gpu/{gt}/act_freq")) else {
            continue;
        };
        let name = clock.label.trim_end_matches(" Clock");
        let value = if clock.current == 0.0 {
            "gated".to_string()
        } else {
            format!("{:.0} MHz", clock.current)
        };
        let rc6 = idx
            .value(&format!("intel_gpu/{gt}/rc6"))
            .map(|p| format!("  RC6 {p:.0}%"))
            .unwrap_or_default();
        lines.push(kv(theme, name, format!("{value}{rc6}"), freq));
    }
    Panel {
        title: "Clocks",
        lines,
    }
}

fn voltage_panel<'a>(
    idx: &Index,
    snapshot: &'a [(SensorId, SensorReading)],
    rows: &[CoreRow],
    theme: &TuiTheme,
) -> Panel<'a> {
    let mut lines = Vec::new();
    let volt = theme.voltage_style();

    let mut rails: Vec<&(SensorId, SensorReading)> = snapshot
        .iter()
        .filter(|(id, r)| id.source == "hwmon" && r.category == SensorCategory::Voltage)
        .collect();
    rails.sort_by(|(a, _), (b, _)| a.natural_cmp(b));
    if !rails.is_empty() {
        lines.push(heading(theme, "Board sense (measured)"));
        for (_, r) in rails {
            lines.push(kv(theme, &r.label, format!("{:.3} V", r.current), volt));
        }
    }

    let mut requested = Vec::new();
    for (kind, name) in [
        (CoreKind::Performance, "P-core VID max"),
        (CoreKind::Efficient, "E-core VID max"),
        (CoreKind::Uniform, "Core VID max"),
    ] {
        if let Some(s) = stats(rows.iter().filter(|r| r.kind == Some(kind)).map(|r| r.vid)) {
            requested.push(kv(theme, name, format!("{:.3} V", s.max), volt));
        }
    }
    for (key, label) in [
        ("cpu/package0/uncore_vid", "Uncore VID"),
        ("mchbar/arl/sa_vid", "SA VID"),
    ] {
        if let Some(v) = idx.value(key) {
            requested.push(kv(theme, label, format!("{v:.3} V"), volt));
        }
    }
    if !requested.is_empty() {
        lines.push(heading(theme, "Requested (VID)"));
        lines.extend(requested);
    }
    Panel {
        title: "Voltage",
        lines,
    }
}

fn power_panel<'a>(idx: &Index, theme: &TuiTheme) -> Panel<'a> {
    let mut lines = Vec::new();
    let power = theme.power_style();
    for (key, label) in [
        ("cpu/rapl/package-0", "Package"),
        ("cpu/rapl/core-0", "IA cores"),
        ("cpu/rapl/uncore-0", "GT"),
    ] {
        if let Some(v) = idx.value(key) {
            lines.push(kv(theme, label, format!("{v:.1} W"), power));
        }
    }
    let mut limits = Vec::new();
    for (key, label) in [
        ("cpu/rapl_limits/package-0-pl1", "PL1"),
        ("cpu/rapl_limits/package-0-pl2", "PL2"),
        ("cpu/rapl_limits/package-0-pl4", "PL4"),
        ("cpu/rapl_limits/package-0-tdp", "Base TDP"),
    ] {
        if let Some(v) = idx.value(key) {
            limits.push(kv(theme, label, format!("{v:.0} W"), theme.info_style()));
        }
    }
    if let Some(a) = idx.value("cpu/package0/vr_current_limit") {
        limits.push(kv(
            theme,
            "VR current limit",
            format!("{a:.1} A"),
            theme.info_style(),
        ));
    }
    if !limits.is_empty() {
        lines.push(heading(theme, "Configured"));
        lines.extend(limits);
    }
    if let Some(p) = idx.value("cpu/package0/c2_residency") {
        lines.push(kv(
            theme,
            "Package C2",
            format!("{p:.1} %"),
            theme.info_style(),
        ));
    }
    Panel {
        title: "Power",
        lines,
    }
}

fn thermal_panel<'a>(
    idx: &Index,
    snapshot: &[(SensorId, SensorReading)],
    rows: &[CoreRow],
    theme: &TuiTheme,
) -> Panel<'a> {
    let mut lines = Vec::new();
    let mut push = |label: &str, t: Option<f64>| {
        if t.is_some() {
            lines.push(kv(
                theme,
                label,
                format!("{} °C", fmt(t, 1)),
                temp_style(theme, t),
            ));
        }
    };
    let pkg = snapshot
        .iter()
        .find(|(id, r)| {
            id.source == "hwmon"
                && id.chip.starts_with("coretemp")
                && r.label.starts_with("Package")
        })
        .map(|(_, r)| r.current);
    push("Package", pkg);
    push(
        "Hottest core",
        stats(rows.iter().map(|r| r.temp)).map(|s| s.max),
    );
    push("IA cores", idx.value("mchbar/arl/ia_temp"));
    push("GT cores", idx.value("mchbar/arl/gt_temp"));
    let peci = snapshot
        .iter()
        .find(|(id, r)| {
            id.source == "hwmon"
                && r.category == SensorCategory::Temperature
                && r.label.contains("PECI")
        })
        .map(|(_, r)| r.current);
    push("Board PECI (avg)", peci);

    if let Some(tj) = idx.value("cpu/package0/tjmax") {
        let offset = idx
            .value("cpu/package0/tcc_offset")
            .filter(|&o| o != 0.0)
            .map(|o| format!(" (TCC -{o:.0})"))
            .unwrap_or_default();
        lines.push(kv(
            theme,
            "TjMAX",
            format!("{tj:.0} °C{offset}"),
            theme.info_style(),
        ));
    }
    if let Some(m) = idx.value("cpu/package0/tjmax_distance") {
        lines.push(kv(
            theme,
            "Pkg to TjMAX",
            format!("{m:.0} °C"),
            theme.info_style(),
        ));
    }
    Panel {
        title: "Thermal",
        lines,
    }
}

fn memory_panel<'a>(idx: &Index, memory: &MemoryInfo, theme: &TuiTheme) -> Panel<'a> {
    let mut lines = Vec::new();
    if let Some(mc) = &memory.controller {
        let t = &mc.timings;
        let rate = mc
            .data_rate_mts
            .map(|r| format!("DDR5-{r}"))
            .unwrap_or_else(|| "DDR5".into());
        let info = theme.info_style();
        lines.push(kv(
            theme,
            "Mode",
            format!("{rate} Gear {} {}N", mc.gear, mc.command_rate),
            info,
        ));
        lines.push(kv(
            theme,
            "tCL-tRCD-tRP-tRAS",
            format!("{}-{}-{}-{}", t.tcl, t.trcd, t.trp, t.tras),
            info,
        ));
        lines.push(kv(
            theme,
            "tRC tRFC tREFI",
            format!("{} {} {}", t.trc, t.trfc, t.trefi),
            info,
        ));
        lines.push(kv(
            theme,
            "tRRD L/S tFAW",
            format!("{}/{} {}", t.trrd_l, t.trrd_s, t.tfaw),
            info,
        ));
        lines.push(kv(
            theme,
            "tCWL tWR tRTP",
            format!("{} {} {}", t.tcwl, t.twr, t.trtp),
            info,
        ));
    }
    for (key, label) in [
        ("uncore/imc/dram_read", "DRAM read"),
        ("uncore/imc/dram_write", "DRAM write"),
        ("uncore/imc/dram_total", "DRAM total"),
    ] {
        if let Some(v) = idx.value(key) {
            lines.push(kv(theme, label, format!("{v:.0} MB/s"), theme.good_style()));
        }
    }
    // DRAM bandwidth is only measured on Arrow Lake-S, where MCHBAR also
    // provides the trained timings; hint only there.
    if memory.controller.is_some() {
        lines.extend(uncore_hint(idx, "uncore/imc/", "uncore_imc_0", theme));
    }
    Panel {
        title: "Memory",
        lines,
    }
}

fn limits_panel<'a>(idx: &Index, theme: &TuiTheme) -> Panel<'a> {
    let mut lines = Vec::new();
    for (key, label) in [
        ("cpu/package0/ia_limit_reasons", "IA"),
        ("cpu/package0/gt_limit_reasons", "GT"),
        ("cpu/package0/ring_limit_reasons", "Ring"),
    ] {
        let Some(word) = idx.value(key).map(|v| v as u64) else {
            continue;
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{label:<5}"), theme.label_style()),
            Span::styled(format!("{word:#010x} "), theme.info_style()),
            Span::styled("live ", theme.muted_style()),
            Span::styled(bit_list(&set_bits(word, 0, 15)), theme.warn_style()),
            Span::styled("  log ", theme.muted_style()),
            Span::styled(bit_list(&set_bits(word, 16, 31)), theme.info_style()),
        ]));
    }
    if let Some(word) = idx.value("cpu/package0/therm_status").map(|v| v as u64) {
        let codes = therm_flag_codes(word);
        lines.push(kv(
            theme,
            "Pkg thermal flags",
            if codes.is_empty() { "-".into() } else { codes },
            theme.warn_style(),
        ));
    }
    if !lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "Raw bits: names differ between generations.",
            theme.muted_style(),
        )));
        lines.push(Line::from(Span::styled(
            "One sample can catch a bit that flickers.",
            theme.muted_style(),
        )));
    }
    Panel {
        title: "Limit reasons",
        lines,
    }
}

fn kind_counts(rows: &[CoreRow]) -> String {
    let p = rows
        .iter()
        .filter(|r| r.kind == Some(CoreKind::Performance))
        .count();
    let e = rows
        .iter()
        .filter(|r| r.kind == Some(CoreKind::Efficient))
        .count();
    if p + e > 0 {
        format!("{p}P + {e}E")
    } else {
        format!("{} CPUs", rows.len())
    }
}

fn core_table_rows<'a>(rows: &[CoreRow], theme: &TuiTheme) -> Vec<Row<'a>> {
    let mut out = Vec::new();
    let freq = theme.info_style();
    let volt = theme.voltage_style();

    // P/E summary rows first, as in turbostat's summary line.
    for kind in [
        CoreKind::Performance,
        CoreKind::Efficient,
        CoreKind::Uniform,
    ] {
        let group: Vec<&CoreRow> = rows.iter().filter(|r| r.kind == Some(kind)).collect();
        if group.is_empty() {
            continue;
        }
        for (which, pick) in [("min", 0usize), ("avg", 1usize), ("max", 2usize)] {
            let col = |f: fn(&CoreRow) -> Option<f64>| {
                stats(group.iter().map(|r| f(r))).map(|s| [s.min, s.avg, s.max][pick])
            };
            let name = match kind {
                CoreKind::Uniform => format!("all {which}"),
                _ => format!("{}-core {which}", kind.short()),
            };
            let temp = col(|r| r.temp);
            out.push(
                Row::new(vec![
                    Cell::from(name).style(theme.accent_style()),
                    Cell::from(fmt(col(|r| r.eff_mhz), 0)).style(freq),
                    Cell::from(fmt(col(|r| r.busy_mhz), 0)).style(freq),
                    Cell::from(fmt(col(|r| r.busy_pct), 1)),
                    Cell::from(fmt(col(|r| r.ratio), 0)),
                    Cell::from(fmt(col(|r| r.vid), 3)).style(volt),
                    Cell::from(fmt(temp, 0)).style(temp_style(theme, temp)),
                    Cell::from(fmt(col(|r| r.margin), 0)),
                    Cell::from(""),
                ])
                .style(Style::default().add_modifier(Modifier::BOLD)),
            );
        }
    }

    for r in rows {
        let kind = r.kind.map(CoreKind::short).unwrap_or("");
        out.push(Row::new(vec![
            Cell::from(format!("{kind:<1} cpu{}", r.cpu)).style(theme.label_style()),
            Cell::from(fmt(r.eff_mhz, 0)).style(freq),
            Cell::from(fmt(r.busy_mhz, 0)).style(freq),
            Cell::from(fmt(r.busy_pct, 1)),
            Cell::from(fmt(r.ratio, 0)),
            Cell::from(fmt(r.vid, 3)).style(volt),
            Cell::from(fmt(r.temp, 0)).style(temp_style(theme, r.temp)),
            Cell::from(fmt(r.margin, 0)),
            Cell::from(r.flags.map(therm_flag_codes).unwrap_or_default()).style(theme.warn_style()),
        ]));
    }
    out
}

fn render_panels(frame: &mut ratatui::Frame, area: Rect, panels: &[Panel<'_>], theme: &TuiTheme) {
    let panels: Vec<&Panel<'_>> = panels.iter().filter(|p| !p.lines.is_empty()).collect();
    if panels.is_empty() {
        return;
    }
    let constraints: Vec<Constraint> = panels
        .iter()
        .map(|p| Constraint::Length(p.lines.len() as u16 + 2))
        .chain(std::iter::once(Constraint::Min(0)))
        .collect();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);
    for (panel, chunk) in panels.iter().zip(chunks.iter()) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_style())
            .title(Span::styled(
                format!(" {} ", panel.title),
                theme.accent_style(),
            ));
        frame.render_widget(Paragraph::new(panel.lines.clone()).block(block), *chunk);
    }
}

fn core_table<'a>(rows: &[CoreRow], theme: &TuiTheme) -> Table<'a> {
    let header = Row::new(vec![
        "CPU", "Eff MHz", "Bzy MHz", "Busy%", "Ratio", "VID", "°C", "ΔTj", "Flags",
    ])
    .style(theme.label_style().add_modifier(Modifier::BOLD));
    let widths = [
        Constraint::Length(10),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(7),
        Constraint::Length(5),
        Constraint::Length(5),
        Constraint::Min(8),
    ];
    Table::new(core_table_rows(rows, theme), widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_style())
                .title(Span::styled(
                    " Cores (VID = requested; lowercase flags = sticky log) ",
                    theme.accent_style(),
                )),
        )
}

/// Rows the core table needs: summary rows, one per CPU, header, borders.
fn core_table_height(rows: &[CoreRow]) -> u16 {
    let kinds = [
        CoreKind::Performance,
        CoreKind::Efficient,
        CoreKind::Uniform,
    ]
    .iter()
    .filter(|k| rows.iter().any(|r| r.kind == Some(**k)))
    .count();
    (kinds * 3 + rows.len() + 3) as u16
}

/// Render panels alternately into two side-by-side columns.
fn render_two_columns<'a>(
    frame: &mut ratatui::Frame,
    area: Rect,
    panels: Vec<Panel<'a>>,
    theme: &TuiTheme,
) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    let (left, right): (Vec<_>, Vec<_>) = panels
        .into_iter()
        .filter(|p| !p.lines.is_empty())
        .enumerate()
        .partition(|(i, _)| i % 2 == 0);
    let strip = |v: Vec<(usize, Panel<'a>)>| v.into_iter().map(|(_, p)| p).collect::<Vec<_>>();
    render_panels(frame, cols[0], &strip(left), theme);
    render_panels(frame, cols[1], &strip(right), theme);
}

pub fn render(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    snapshot: &[(SensorId, SensorReading)],
    memory_info: &MemoryInfo,
    sys: &SystemSummary,
    elapsed_str: &str,
    theme: &TuiTheme,
) -> io::Result<()> {
    let rows = core_rows(snapshot);
    let idx = Index::new(snapshot);

    terminal.draw(|frame| {
        let size = frame.area();
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(size);

        let header = Paragraph::new(format!(
            " CPU Detail | {} | {} | {elapsed_str}",
            sys.cpu_model,
            kind_counts(&rows)
        ))
        .style(theme.accent_style())
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(theme.border_style()),
        );
        frame.render_widget(header, outer[0]);

        let status = Paragraph::new(
            " q: quit | d: dashboard | m: DIMM | p: this view | t: theme | /: search",
        )
        .style(theme.status_style());
        frame.render_widget(status, outer[2]);

        let main = outer[1];
        if rows.is_empty() {
            let msg = Paragraph::new(
                " No per-core MSR data: run as root (or with CAP_SYS_RAWIO) with the msr module loaded",
            )
            .style(theme.label_style());
            frame.render_widget(msg, main);
            return;
        }

        let mut panels = vec![
            clocks_panel(&idx, &rows, theme),
            voltage_panel(&idx, snapshot, &rows, theme),
            power_panel(&idx, theme),
            thermal_panel(&idx, snapshot, &rows, theme),
            memory_panel(&idx, memory_info, theme),
            limits_panel(&idx, theme),
        ];

        // Wide: table with Thermal/Memory/Limits under it, and Clocks,
        // Voltage and Power in a side column. Narrow: everything stacked.
        let (left_col, side) = if main.width >= 130 {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(80), Constraint::Length(50)])
                .split(main);
            (cols[0], Some(cols[1]))
        } else {
            (main, None)
        };
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(core_table_height(&rows)),
                Constraint::Min(0),
            ])
            .split(left_col);
        frame.render_widget(core_table(&rows, theme), split[0]);

        if let Some(side) = side {
            let below = panels.split_off(3);
            render_panels(frame, side, &panels, theme);
            render_two_columns(frame, split[1], below, theme);
        } else {
            render_two_columns(frame, split[1], panels, theme);
        }
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::sensor::SensorUnit;

    fn reading(
        source: &str,
        chip: &str,
        sensor: &str,
        label: &str,
        value: f64,
    ) -> (SensorId, SensorReading) {
        (
            SensorId {
                source: source.into(),
                chip: chip.into(),
                sensor: sensor.into(),
            },
            SensorReading::new(
                label.into(),
                value,
                SensorUnit::Unitless,
                SensorCategory::Other,
            ),
        )
    }

    fn sample() -> Vec<(SensorId, SensorReading)> {
        vec![
            reading("cpu", "package0", "tjmax", "TjMAX", 105.0),
            reading("cpu", "vid", "cpu0", "P-Core 0 VID", 1.386),
            reading("cpu", "vid", "cpu1", "P-Core 1 VID", 1.271),
            reading("cpu", "vid", "cpu8", "E-Core 8 VID", 0.775),
            reading("cpu", "tjmax_distance", "cpu0", "P-Core 0 Distance", 71.0),
            reading("cpu", "eff_clock", "cpu8", "E-Core 8 Effective Clock", 59.2),
            reading(
                "cpu",
                "therm_status",
                "cpu1",
                "P-Core 1 Thermal Status",
                8192.0,
            ),
        ]
    }

    #[test]
    fn test_core_rows_sorted_with_kinds() {
        let rows = core_rows(&sample());
        assert_eq!(rows.iter().map(|r| r.cpu).collect::<Vec<_>>(), [0, 1, 8]);
        assert_eq!(rows[0].kind, Some(CoreKind::Performance));
        assert_eq!(rows[2].kind, Some(CoreKind::Efficient));
        // Temperature is TjMAX minus the DTS readout
        assert_eq!(rows[0].temp, Some(34.0));
        assert_eq!(rows[1].temp, None);
        assert_eq!(rows[1].flags, Some(0x2000));
        assert_eq!(rows[2].eff_mhz, Some(59.2));
    }

    #[test]
    fn test_stats_skip_missing() {
        let s = stats([Some(1.0), None, Some(3.0)].into_iter()).unwrap();
        assert_eq!((s.min, s.avg, s.max), (1.0, 2.0, 3.0));
        assert!(stats([None, None].into_iter()).is_none());
    }

    #[test]
    fn test_therm_flag_codes() {
        // bit 13: current-limit log only
        assert_eq!(therm_flag_codes(0x2000), "cl");
        // bit 0 live thermal + bit 1 log, bit 10 power limit live
        assert_eq!(therm_flag_codes(0x0403), "TH th PL");
        assert_eq!(therm_flag_codes(0), "");
    }

    #[test]
    fn test_limit_reason_bits() {
        // IA word recorded at idle: logged bits 17, 22, 23, 24, 28
        let word = 0x11C2_0000;
        assert!(set_bits(word, 0, 15).is_empty());
        assert_eq!(set_bits(word, 16, 31), vec![17, 22, 23, 24, 28]);
        assert_eq!(bit_list(&[]), "-");
    }

    #[test]
    fn test_kind_counts() {
        let rows = core_rows(&sample());
        assert_eq!(kind_counts(&rows), "2P + 1E");
    }
}
