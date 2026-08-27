//! Pure application state -- no wasm-bindgen types in here on purpose. This
//! is the "shared data without FFI" half of the architecture: everything
//! that arrives over the two WebSockets and the chat SSE stream lands here
//! as plain Rust, gets aggregated here, and is read directly by render.rs
//! to write the DOM. It never round-trips through a JS object.

use std::collections::HashMap;

/// Same window this project's `ui/cupti-activity.js` uses -- 3s on CUPTI's
/// own device clock, not client arrival time (bursty/delayed delivery means
/// a client-time window can span a different, longer real interval; see
/// that file's own extensive comment on why this bit it in testing).
pub const ROLLING_WINDOW_NS: i64 = 3_000_000_000;

/// How many distinct kernel-name rows the bar list shows before folding the
/// rest into a "+N more" footnote.
pub const MAX_ROWS: usize = 15;

/// How often the render loop ticks, matching `ui/trace.js`'s proven
/// GPU_REFRESH_INTERVAL_MS cadence.
pub const RENDER_INTERVAL_MS: i32 = 150;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Phase {
    #[default]
    Response,
    Thinking,
}

#[derive(Clone, Debug, Default)]
pub struct KernelStat {
    pub thinking: u32,
    pub response: u32,
}

impl KernelStat {
    pub fn total(&self) -> u32 {
        self.thinking + self.response
    }
}

/// What render.rs last actually wrote to the DOM, so it only touches a node
/// again when the value genuinely changed -- this diff-before-write is what
/// cuts down real DOM/FFI traffic, not a novel memory format.
#[derive(Default)]
pub struct RenderCache {
    pub now_executing: Option<String>,
    pub rows: Vec<(String, u32, u32)>,
    pub overflow_count: usize,
    pub busy_percent: Option<i32>,
    pub busy_caption: Option<String>,
    pub prompt_tokens: Option<u32>,
    pub semantic_connected: Option<bool>,
    pub cupti_connected: Option<bool>,
}

#[derive(Default)]
pub struct AppState {
    pub kernel_stats: HashMap<String, KernelStat>,
    pub phase: Phase,
    pub now_executing: Option<String>,

    // Rolling GPU-busy bookkeeping -- ported verbatim from
    // ui/cupti-activity.js's mergedBusyNs/getRollingBusy. Session-scoped,
    // NOT reset per query (matches that file's own design: it's "device
    // busy time," not a per-query stat).
    pub recent_launches: Vec<(i64, i64)>, // (startNs, endNs)
    pub latest_end_ns: i64,
    pub first_start_ns: Option<i64>,

    pub prompt_tokens: Option<u32>,
    pub semantic_connected: bool,
    pub cupti_connected: bool,

    pub reply_text: String,
    pub chat_busy: bool,

    pub render_cache: RenderCache,
}

impl AppState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called once per new chat turn -- zeroes the per-query kernel tally
    /// and phase, exactly like `resetCuptiQueryCounters` in
    /// `ui/cupti-activity.js`. Rolling-busy data is left alone (session-
    /// wide signal).
    pub fn reset_query(&mut self) {
        self.kernel_stats.clear();
        self.phase = Phase::Response;
        self.prompt_tokens = None;
        self.reply_text.clear();
    }

    pub fn set_thinking(&mut self, is_thinking: bool) {
        self.phase = if is_thinking { Phase::Thinking } else { Phase::Response };
    }

    /// One real classified... well, unclassified: this is the generic-feed
    /// design, so there is no classify() at all -- every real launch is
    /// bucketed straight under its own raw (already-demangled) kernel name.
    pub fn record_launch(&mut self, name: &str, start_ns: i64, end_ns: i64) {
        self.now_executing = Some(name.to_string());

        let stat = self.kernel_stats.entry(name.to_string()).or_default();
        match self.phase {
            Phase::Thinking => stat.thinking += 1,
            Phase::Response => stat.response += 1,
        }

        if end_ns > start_ns {
            self.recent_launches.push((start_ns, end_ns));
            if end_ns > self.latest_end_ns {
                self.latest_end_ns = end_ns;
            }
            if self.first_start_ns.is_none() {
                self.first_start_ns = Some(start_ns);
            }
        }
    }

    /// Returns (busy_ns, window_ns, launch_count) -- direct port of
    /// getRollingBusy()/mergedBusyNs() from ui/cupti-activity.js.
    pub fn rolling_busy(&mut self) -> (i64, i64, usize) {
        if self.latest_end_ns == 0 {
            return (0, 0, 0);
        }
        let first_start = self.first_start_ns.unwrap_or(0);
        let cutoff = (self.latest_end_ns - ROLLING_WINDOW_NS).max(first_start);
        self.recent_launches.retain(|(_, end_ns)| *end_ns >= cutoff);
        let window_ns = self.latest_end_ns - cutoff;

        let mut intervals: Vec<(i64, i64)> = self
            .recent_launches
            .iter()
            .map(|(start_ns, end_ns)| (start_ns.max(&cutoff).to_owned(), *end_ns))
            .collect();
        let busy_ns = merged_busy_ns(&mut intervals);

        (busy_ns, window_ns, self.recent_launches.len())
    }

    /// Top rows by total count descending, capped at MAX_ROWS; returns the
    /// visible rows plus how many distinct names were left out.
    pub fn top_rows(&self, cap: usize) -> (Vec<(&str, &KernelStat)>, usize) {
        let mut all: Vec<(&str, &KernelStat)> =
            self.kernel_stats.iter().map(|(k, v)| (k.as_str(), v)).collect();
        all.sort_by(|a, b| b.1.total().cmp(&a.1.total()).then_with(|| a.0.cmp(b.0)));
        let overflow = all.len().saturating_sub(cap);
        all.truncate(cap);
        (all, overflow)
    }
}

fn merged_busy_ns(intervals: &mut [(i64, i64)]) -> i64 {
    if intervals.is_empty() {
        return 0;
    }
    intervals.sort_by_key(|(start, _)| *start);
    let mut busy = 0i64;
    let (mut cur_start, mut cur_end) = intervals[0];
    for &(start, end) in &intervals[1..] {
        if start <= cur_end {
            cur_end = cur_end.max(end);
        } else {
            busy += cur_end - cur_start;
            cur_start = start;
            cur_end = end;
        }
    }
    busy += cur_end - cur_start;
    busy
}
