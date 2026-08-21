//! Cold-path, bounded offline analysis for a completed CUPTI capture.
//!
//! This binary deliberately uses `std` for filesystem I/O. It is never loaded into
//! vLLM and is not part of the inference hot path. The live semantic transport is
//! implemented by the `no_std` `gpu-observer-core` crate; CUPTI capture uses a
//! preallocated fixed buffer pool. Every input and record vector here has an explicit
//! upper bound so malformed traces fail closed instead of growing without limit.

use gpu_observer_collector::join_b_trace::{load_semantic, SemanticTrace};
use std::{
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

const MAX_ACTIVITY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_API_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SUMMARY_BYTES: u64 = 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;
const FAMILY_COUNT: usize = 10;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelFamily {
    LayerNorm = 0,
    Gemm = 1,
    QkNorm = 2,
    Rotary = 3,
    KvCache = 4,
    Attention = 5,
    Activation = 6,
    Sampling = 7,
    Copy = 8,
    Other = 9,
}

impl KernelFamily {
    const ALL: [Self; FAMILY_COUNT] = [
        Self::LayerNorm,
        Self::Gemm,
        Self::QkNorm,
        Self::Rotary,
        Self::KvCache,
        Self::Attention,
        Self::Activation,
        Self::Sampling,
        Self::Copy,
        Self::Other,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::LayerNorm => "layer_norm",
            Self::Gemm => "gemm_projection",
            Self::QkNorm => "qk_norm",
            Self::Rotary => "rotary",
            Self::KvCache => "kv_cache_write",
            Self::Attention => "attention",
            Self::Activation => "activation",
            Self::Sampling => "sampling",
            Self::Copy => "copy",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct KernelActivity {
    start_ns: u64,
    end_ns: u64,
    stream: u32,
    correlation: u32,
    graph_node_id: u64,
    graph_id: u32,
    family: KernelFamily,
}

#[derive(Clone, Copy, Debug)]
struct ApiActivity {
    start_ns: u64,
    end_ns: u64,
    process: u32,
    correlation: u32,
    return_value: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct ClockPair {
    cupti_ns: u64,
    monotonic_ns: u64,
    uncertainty_ns: u64,
}

#[derive(Clone, Copy, Debug)]
struct ClockMap {
    start: ClockPair,
    end: ClockPair,
}

#[derive(Debug, Default)]
struct CuptiQuality {
    register_result: i32,
    enable_result: i32,
    runtime_enable_result: i32,
    buffers_requested: u64,
    buffers_completed: u64,
    buffer_exhaustions: u64,
    records_seen: u64,
    target_records: u64,
    invalid_records: u64,
    dropped_records: u64,
    api_records: u64,
    api_drops: u64,
    driver_enable_result: i32,
}

#[derive(Clone, Copy, Debug)]
struct Interval {
    start_ns: u64,
    end_ns: u64,
    api_end_ns: u64,
    family: KernelFamily,
    graph: bool,
}

#[derive(Default)]
struct StepAggregate {
    intervals: Vec<Interval>,
    streams: Vec<u32>,
    api_first_ns: u64,
    api_last_ns: u64,
}

#[derive(Default)]
struct Totals {
    kernels: u64,
    graph_kernels: u64,
    ordinary_kernels: u64,
    kernel_sum_ns: u64,
    per_step_busy_sum_ns: u64,
    family_counts: [u64; FAMILY_COUNT],
    family_duration_ns: [u64; FAMILY_COUNT],
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = PathBuf::from(args.next().unwrap_or_default());
    let usage = || {
        format!(
            "usage: {} SEMANTIC.bin ACTIVITIES.tsv API.tsv SUMMARY.tsv STEPS.tsv FAMILIES.tsv",
            program.display()
        )
    };
    let semantic_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let activity_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let api_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let summary_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let steps_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let families_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    if args.next().is_some() {
        return Err(usage().into());
    }

    let semantic = load_semantic(&semantic_path)?;
    let mut kernels = load_kernels(&activity_path)?;
    let mut apis = load_apis(&api_path)?;
    let (quality, clock) = load_summary(&summary_path)?;
    validate_inputs(&semantic, &kernels, &apis, &quality, &clock)?;

    kernels.sort_unstable_by_key(|kernel| (kernel.start_ns, kernel.end_ns));
    apis.sort_unstable_by_key(|api| api.correlation);
    if apis
        .windows(2)
        .any(|pair| pair[0].correlation == pair[1].correlation)
    {
        return Err("CUPTI api correlation IDs are not unique".into());
    }

    let packed_timeline = packed_timeline(&semantic)?;
    let mut aggregates = zeroed_aggregates(semantic.steps.len())?;
    let mut all_intervals = Vec::new();
    all_intervals.try_reserve_exact(kernels.len())?;
    let mut unmatched_correlations = 0_u64;
    let mut unassigned_kernels = 0_u64;
    let mut api_pid_mismatches = 0_u64;
    let mut invalid_intervals = 0_u64;

    for kernel in &kernels {
        let api = apis
            .binary_search_by_key(&kernel.correlation, |api| api.correlation)
            .ok()
            .map(|index| apis[index]);
        let Some(api) = api else {
            unmatched_correlations += 1;
            continue;
        };
        let api_start_ns = clock.normalize(api.start_ns)?;
        let api_end_ns = clock.normalize(api.end_ns)?;
        let kernel_start_ns = clock.normalize(kernel.start_ns)?;
        let kernel_end_ns = clock.normalize(kernel.end_ns)?;
        if api.return_value != 0 || api_end_ns < api_start_ns || kernel_end_ns <= kernel_start_ns {
            invalid_intervals += 1;
            continue;
        }
        let Some(step_index) = step_for_packed_timestamp(api_start_ns, &semantic, &packed_timeline)
        else {
            unassigned_kernels += 1;
            continue;
        };
        let step = &semantic.steps[step_index];
        if api.process != step.begin.pid {
            api_pid_mismatches += 1;
            continue;
        }
        let interval = Interval {
            start_ns: kernel_start_ns,
            end_ns: kernel_end_ns,
            api_end_ns,
            family: kernel.family,
            graph: kernel.graph_node_id != 0 || kernel.graph_id != 0,
        };
        let aggregate = &mut aggregates[step_index];
        aggregate.intervals.try_reserve(1)?;
        aggregate.intervals.push(interval);
        aggregate.api_first_ns = if aggregate.api_first_ns == 0 {
            api_start_ns
        } else {
            aggregate.api_first_ns.min(api_start_ns)
        };
        aggregate.api_last_ns = aggregate.api_last_ns.max(api_end_ns);
        if !aggregate.streams.contains(&kernel.stream) {
            aggregate.streams.try_reserve(1)?;
            aggregate.streams.push(kernel.stream);
        }
        all_intervals.push((kernel_start_ns, kernel_end_ns));
    }

    let mut steps_output = BufWriter::new(File::create(&steps_path)?);
    writeln!(
        steps_output,
        "step_id\tphase\tscheduled_tokens\tprefill_tokens\tdecode_tokens\trequests\tqueue_depth\tsemantic_wall_ns\tschedule_to_pack_ns\tpack_to_first_api_ns\tapi_endpoint_span_ns\tfirst_gpu_start_minus_api_end_ns\tkernel_count\tgraph_kernel_count\tordinary_kernel_count\tstreams\tkernel_sum_ns\tgpu_busy_union_ns\tgpu_span_ns\tgpu_gap_within_span_ns\tlast_gpu_to_step_end_ns\treordered_positions"
    )?;
    let mut families_output = BufWriter::new(File::create(&families_path)?);
    writeln!(
        families_output,
        "step_id\tphase\tfamily\tkernel_count\tkernel_duration_sum_ns\tevidence"
    )?;

    let mut totals = Totals::default();
    let mut empty_steps = 0_u64;
    let mut gpu_outside_semantic_steps = 0_u64;
    let mut negative_api_overlap_intervals = 0_u64;
    for (step_index, step) in semantic.steps.iter().enumerate() {
        let aggregate = &mut aggregates[step_index];
        aggregate
            .intervals
            .sort_unstable_by_key(|interval| (interval.start_ns, interval.end_ns));
        if aggregate.intervals.is_empty() {
            empty_steps += 1;
            continue;
        }
        let packed_ns = step
            .packed_begin
            .ok_or("step is missing authoritative packed-layout begin")?
            .timestamp_ns;
        if packed_ns < step.begin.timestamp_ns || step.end_timestamp_ns < packed_ns {
            return Err("semantic schedule/pack/end ordering is invalid".into());
        }
        let phase = phase_label(step.begin.prefill_tokens, step.begin.decode_tokens)?;
        let kernel_sum_ns = duration_sum(&aggregate.intervals)?;
        let busy_union_ns = interval_union_from_records(&aggregate.intervals)?;
        let first = aggregate
            .intervals
            .first()
            .ok_or("step lost its first interval")?;
        let last_gpu_end_ns = aggregate
            .intervals
            .iter()
            .map(|interval| interval.end_ns)
            .max()
            .ok_or("step lost its final interval")?;
        let gpu_span_ns = last_gpu_end_ns - first.start_ns;
        let gpu_gap_ns = gpu_span_ns.saturating_sub(busy_union_ns);
        let graph_kernel_count = aggregate
            .intervals
            .iter()
            .filter(|interval| interval.graph)
            .count();
        let ordinary_kernel_count = aggregate.intervals.len() - graph_kernel_count;
        let first_gpu_start_minus_api_end_ns = signed_delta(first.start_ns, first.api_end_ns);
        negative_api_overlap_intervals += u64::from(first_gpu_start_minus_api_end_ns < 0);
        gpu_outside_semantic_steps += u64::from(
            first.start_ns < step.begin.timestamp_ns || last_gpu_end_ns > step.end_timestamp_ns,
        );
        let semantic_wall_ns = step.end_timestamp_ns - step.begin.timestamp_ns;
        writeln!(
            steps_output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            step.begin.step_id,
            phase,
            step.begin.scheduled_tokens,
            step.begin.prefill_tokens,
            step.begin.decode_tokens,
            step.owner_count,
            step.begin.queue_depth,
            semantic_wall_ns,
            packed_ns - step.begin.timestamp_ns,
            aggregate.api_first_ns.saturating_sub(packed_ns),
            aggregate.api_last_ns.saturating_sub(aggregate.api_first_ns),
            first_gpu_start_minus_api_end_ns,
            aggregate.intervals.len(),
            graph_kernel_count,
            ordinary_kernel_count,
            aggregate.streams.len(),
            kernel_sum_ns,
            busy_union_ns,
            gpu_span_ns,
            gpu_gap_ns,
            signed_delta(step.end_timestamp_ns, last_gpu_end_ns),
            step.scheduler_order_mismatches,
        )?;

        let mut family_counts = [0_u64; FAMILY_COUNT];
        let mut family_durations = [0_u64; FAMILY_COUNT];
        for interval in &aggregate.intervals {
            let family_index = interval.family as usize;
            family_counts[family_index] += 1;
            family_durations[family_index] = family_durations[family_index]
                .checked_add(interval.end_ns - interval.start_ns)
                .ok_or("family duration overflow")?;
        }
        for family in KernelFamily::ALL {
            let index = family as usize;
            if family_counts[index] == 0 {
                continue;
            }
            writeln!(
                families_output,
                "{}\t{}\t{}\t{}\t{}\tkernel_symbol_pattern_reconstruction",
                step.begin.step_id,
                phase,
                family.label(),
                family_counts[index],
                family_durations[index],
            )?;
            totals.family_counts[index] += family_counts[index];
            totals.family_duration_ns[index] = totals.family_duration_ns[index]
                .checked_add(family_durations[index])
                .ok_or("total family duration overflow")?;
        }
        totals.kernels += u64::try_from(aggregate.intervals.len())?;
        totals.graph_kernels += u64::try_from(graph_kernel_count)?;
        totals.ordinary_kernels += u64::try_from(ordinary_kernel_count)?;
        totals.kernel_sum_ns = totals
            .kernel_sum_ns
            .checked_add(kernel_sum_ns)
            .ok_or("total kernel sum overflow")?;
        totals.per_step_busy_sum_ns = totals
            .per_step_busy_sum_ns
            .checked_add(busy_union_ns)
            .ok_or("per-step busy sum overflow")?;
    }
    steps_output.flush()?;
    families_output.flush()?;

    all_intervals.sort_unstable_by_key(|interval| (interval.0, interval.1));
    let global_busy_union_ns = interval_union(&all_intervals)?;
    let clock_offset_drift_ns = clock.end.offset_ns() - clock.start.offset_ns();
    let accounting_mismatch =
        u64::from(quality.records_seen != quality.target_records + quality.api_records);
    let passed = semantic.loss_markers == 0
        && semantic.packed_steps == semantic.steps.len() as u64
        && quality.target_records == kernels.len() as u64
        && totals.kernels == kernels.len() as u64
        && unmatched_correlations == 0
        && unassigned_kernels == 0
        && api_pid_mismatches == 0
        && invalid_intervals == 0
        && empty_steps == 0
        && gpu_outside_semantic_steps == 0
        && accounting_mismatch == 0;

    for family in KernelFamily::ALL {
        let index = family as usize;
        println!(
            "family={} kernels={} kernel_duration_sum_ns={} evidence=kernel_symbol_pattern_reconstruction",
            family.label(),
            totals.family_counts[index],
            totals.family_duration_ns[index],
        );
    }
    println!(
        "summary status={} semantic_records={} semantic_steps={} semantic_loss={} packed_steps={} max_steps_in_flight={} scheduler_order_mismatch_steps={} cupti_records_seen={} kernel_records={} api_records={} runtime_enable_result={} driver_enable_result={} buffers_requested={} buffers_completed={} buffer_exhaustions={} cupti_invalid_records={} cupti_output_drops={} api_drops={} unmatched_correlations={} unassigned_kernels={} api_pid_mismatches={} invalid_intervals={} empty_steps={} gpu_outside_semantic_steps={} negative_api_overlap_intervals={} accounting_mismatch={} graph_kernels={} ordinary_kernels={} streams_observed={} kernel_sum_ns={} per_step_busy_sum_ns={} global_busy_union_ns={} overlap_ns={} clock_start_uncertainty_ns={} clock_end_uncertainty_ns={} clock_offset_drift_ns={} assignment=runtime_or_driver_correlation_then_authoritative_packed_window timing=normalized_CUPTI_actual_intervals family_evidence=kernel_symbol_pattern_reconstruction",
        if passed { "PASS" } else { "FAIL" },
        semantic.records,
        semantic.steps.len(),
        semantic.loss_markers,
        semantic.packed_steps,
        semantic.max_steps_in_flight,
        semantic.scheduler_order_mismatch_steps,
        quality.records_seen,
        kernels.len(),
        quality.api_records,
        quality.runtime_enable_result,
        quality.driver_enable_result,
        quality.buffers_requested,
        quality.buffers_completed,
        quality.buffer_exhaustions,
        quality.invalid_records,
        quality.dropped_records,
        quality.api_drops,
        unmatched_correlations,
        unassigned_kernels,
        api_pid_mismatches,
        invalid_intervals,
        empty_steps,
        gpu_outside_semantic_steps,
        negative_api_overlap_intervals,
        accounting_mismatch,
        totals.graph_kernels,
        totals.ordinary_kernels,
        aggregates
            .iter()
            .flat_map(|aggregate| aggregate.streams.iter().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        totals.kernel_sum_ns,
        totals.per_step_busy_sum_ns,
        global_busy_union_ns,
        totals.kernel_sum_ns.saturating_sub(global_busy_union_ns),
        clock.start.uncertainty_ns,
        clock.end.uncertainty_ns,
        clock_offset_drift_ns,
    );
    if !passed {
        return Err("full-step CUPTI quality gate failed".into());
    }
    Ok(())
}

fn phase_label(prefill_tokens: u32, decode_tokens: u32) -> Result<&'static str, Box<dyn Error>> {
    match (prefill_tokens != 0, decode_tokens != 0) {
        (true, false) => Ok("prefill"),
        (false, true) => Ok("decode"),
        (true, true) => Ok("mixed"),
        (false, false) => Err("empty engine step".into()),
    }
}

fn classify_kernel(name: &str) -> KernelFamily {
    if name.contains("reshape_and_cache") || name.contains("cache_kernel") {
        KernelFamily::KvCache
    } else if name.contains("flash_fwd") || name.contains("attention") {
        KernelFamily::Attention
    } else if name.contains("triton_red_fused_3")
        || name.contains("q_norm")
        || name.contains("k_norm")
    {
        KernelFamily::QkNorm
    } else if name.contains("triton_poi_fused_4") || name.contains("rotary") {
        KernelFamily::Rotary
    } else if name.contains("mul_silu") || name.contains("SiluAndMul") {
        KernelFamily::Activation
    } else if name.contains("rsqrt") || name.contains("rms_norm") {
        KernelFamily::LayerNorm
    } else if name.contains("gemvx")
        || name.contains("gemm")
        || name.contains("cutlass")
        || name.contains("nvjet")
        || name.contains("cublas")
    {
        KernelFamily::Gemm
    } else if name.contains("radix_sort")
        || name.contains("SoftMax")
        || name.contains("softmax")
        || name.contains("distribution_")
        || name.contains("scatter_gather")
        || name.contains("ArgMax")
        || name.contains("reduce_kernel")
        || name.contains("DeviceScan")
        || name.contains("masked_fill")
    {
        KernelFamily::Sampling
    } else if name.contains("memcpy") || name.contains("copy_kernel") {
        KernelFamily::Copy
    } else {
        KernelFamily::Other
    }
}

fn validate_inputs(
    semantic: &SemanticTrace,
    kernels: &[KernelActivity],
    apis: &[ApiActivity],
    quality: &CuptiQuality,
    clock: &ClockMap,
) -> Result<(), Box<dyn Error>> {
    if semantic.steps.is_empty()
        || semantic.loss_markers != 0
        || semantic.packed_steps != semantic.steps.len() as u64
    {
        return Err("semantic packed-layout gate failed".into());
    }
    if kernels.is_empty() || apis.is_empty() {
        return Err("CUPTI kernel or api trace is empty".into());
    }
    if quality.register_result != 0
        || quality.enable_result != 0
        || quality.runtime_enable_result != 0
        || quality.driver_enable_result != 0
        || quality.buffer_exhaustions != 0
        || quality.invalid_records != 0
        || quality.dropped_records != 0
        || quality.api_drops != 0
        || quality.target_records != kernels.len() as u64
        || quality.api_records != apis.len() as u64
        || quality.buffers_requested != quality.buffers_completed
    {
        return Err(format!("CUPTI quality gate failed: {quality:?}").into());
    }
    if clock.start.cupti_ns == 0
        || clock.end.cupti_ns <= clock.start.cupti_ns
        || clock.end.monotonic_ns <= clock.start.monotonic_ns
    {
        return Err("CUPTI clock calibration is invalid".into());
    }
    Ok(())
}

impl ClockPair {
    fn offset_ns(self) -> i128 {
        i128::from(self.monotonic_ns) - i128::from(self.cupti_ns)
    }
}

impl ClockMap {
    fn normalize(self, timestamp_ns: u64) -> Result<u64, Box<dyn Error>> {
        if timestamp_ns < self.start.cupti_ns || timestamp_ns > self.end.cupti_ns {
            return Err("CUPTI activity timestamp lies outside calibration interval".into());
        }
        let x = u128::from(timestamp_ns - self.start.cupti_ns);
        let x_span = u128::from(self.end.cupti_ns - self.start.cupti_ns);
        let y_span = u128::from(self.end.monotonic_ns - self.start.monotonic_ns);
        let scaled = x
            .checked_mul(y_span)
            .ok_or("clock interpolation overflow")?
            / x_span;
        self.start
            .monotonic_ns
            .checked_add(u64::try_from(scaled)?)
            .ok_or_else(|| "normalized timestamp overflow".into())
    }
}

fn signed_delta(left: u64, right: u64) -> i128 {
    i128::from(left) - i128::from(right)
}

fn packed_timeline(semantic: &SemanticTrace) -> Result<Vec<(u64, usize)>, Box<dyn Error>> {
    let mut timeline = Vec::new();
    timeline.try_reserve_exact(semantic.steps.len())?;
    for (index, step) in semantic.steps.iter().enumerate() {
        timeline.push((
            step.packed_begin
                .ok_or("step is missing packed layout")?
                .timestamp_ns,
            index,
        ));
    }
    timeline.sort_unstable_by_key(|entry| entry.0);
    if timeline.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err("packed timestamps are not strictly increasing".into());
    }
    Ok(timeline)
}

fn step_for_packed_timestamp(
    timestamp_ns: u64,
    semantic: &SemanticTrace,
    timeline: &[(u64, usize)],
) -> Option<usize> {
    let position = timeline.partition_point(|entry| entry.0 <= timestamp_ns);
    if position == 0 {
        return None;
    }
    let index = timeline[position - 1].1;
    (timestamp_ns <= semantic.steps[index].end_timestamp_ns).then_some(index)
}

fn duration_sum(intervals: &[Interval]) -> Result<u64, Box<dyn Error>> {
    intervals.iter().try_fold(0_u64, |sum, interval| {
        sum.checked_add(interval.end_ns - interval.start_ns)
            .ok_or_else(|| "kernel duration sum overflow".into())
    })
}

fn interval_union_from_records(intervals: &[Interval]) -> Result<u64, Box<dyn Error>> {
    let mut flat = Vec::new();
    flat.try_reserve_exact(intervals.len())?;
    flat.extend(
        intervals
            .iter()
            .map(|interval| (interval.start_ns, interval.end_ns)),
    );
    interval_union(&flat)
}

fn interval_union(intervals: &[(u64, u64)]) -> Result<u64, Box<dyn Error>> {
    let Some(&(mut start, mut end)) = intervals.first() else {
        return Ok(0);
    };
    if end <= start {
        return Err("invalid interval".into());
    }
    let mut total = 0_u64;
    for &(next_start, next_end) in &intervals[1..] {
        if next_end <= next_start || next_start < start {
            return Err("unsorted or invalid interval".into());
        }
        if next_start <= end {
            end = end.max(next_end);
        } else {
            total = total
                .checked_add(end - start)
                .ok_or("interval union overflow")?;
            start = next_start;
            end = next_end;
        }
    }
    total
        .checked_add(end - start)
        .ok_or_else(|| "interval union overflow".into())
}

fn zeroed_aggregates(len: usize) -> Result<Vec<StepAggregate>, Box<dyn Error>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len)?;
    values.resize_with(len, StepAggregate::default);
    Ok(values)
}

fn load_kernels(path: &Path) -> Result<Vec<KernelActivity>, Box<dyn Error>> {
    let file = bounded_file(path, MAX_ACTIVITY_BYTES)?;
    let mut output = Vec::new();
    let mut format = false;
    let mut header = false;
    for (line_number, line) in BufReader::with_capacity(1024 * 1024, file)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.starts_with("# format=GOCUPTI01 ") {
            format = true;
            if !line.contains("target=<all>") {
                return Err(
                    "CUPTI activity trace is filtered; full-step analysis requires target=<all>"
                        .into(),
                );
            }
            continue;
        }
        if line.starts_with("start_ns\tend_ns\tdevice\t") {
            header = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if output.len() == MAX_RECORDS {
            return Err("CUPTI kernel trace exceeds collector bound".into());
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 18 {
            return Err(format!("kernel row {} is malformed", line_number + 1).into());
        }
        let activity = KernelActivity {
            start_ns: fields[0].parse()?,
            end_ns: fields[1].parse()?,
            stream: fields[4].parse()?,
            correlation: fields[5].parse()?,
            graph_node_id: fields[7].parse()?,
            graph_id: fields[8].parse()?,
            family: classify_kernel(fields[17]),
        };
        if activity.start_ns == 0
            || activity.end_ns <= activity.start_ns
            || activity.correlation == 0
            || fields[17].is_empty()
        {
            return Err(format!("kernel row {} is invalid", line_number + 1).into());
        }
        output.try_reserve(1)?;
        output.push(activity);
    }
    if !format || !header {
        return Err("CUPTI kernel trace lacks its versioned header".into());
    }
    Ok(output)
}

fn load_apis(path: &Path) -> Result<Vec<ApiActivity>, Box<dyn Error>> {
    let file = bounded_file(path, MAX_API_BYTES)?;
    let mut output = Vec::new();
    let mut format = false;
    let mut header = false;
    for (line_number, line) in BufReader::with_capacity(1024 * 1024, file)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.starts_with("# format=GOCUPTI_RUNTIME01 ")
            || line.starts_with("# format=GOCUPTI_API01 ")
        {
            format = true;
            continue;
        }
        if line.starts_with("start_ns\tend_ns\tprocess\t") {
            header = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if output.len() == MAX_RECORDS {
            return Err("CUPTI api trace exceeds collector bound".into());
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 7 {
            return Err(format!("api row {} is malformed", line_number + 1).into());
        }
        let activity = ApiActivity {
            start_ns: fields[0].parse()?,
            end_ns: fields[1].parse()?,
            process: fields[2].parse()?,
            correlation: fields[4].parse()?,
            return_value: fields[6].parse()?,
        };
        if activity.start_ns == 0
            || activity.end_ns < activity.start_ns
            || activity.correlation == 0
        {
            return Err(format!("api row {} is invalid", line_number + 1).into());
        }
        output.try_reserve(1)?;
        output.push(activity);
    }
    if !format || !header {
        return Err("CUPTI api trace lacks its versioned header".into());
    }
    Ok(output)
}

fn load_summary(path: &Path) -> Result<(CuptiQuality, ClockMap), Box<dyn Error>> {
    let file = bounded_file(path, MAX_SUMMARY_BYTES)?;
    let lines: Vec<String> = BufReader::new(file).lines().collect::<Result<_, _>>()?;
    if lines
        .first()
        .is_none_or(|line| !line.starts_with("# format=GOCUPTI01 "))
    {
        return Err("CUPTI summary lacks its versioned header".into());
    }
    let values: Vec<&str> = lines
        .get(2)
        .ok_or("CUPTI summary is truncated")?
        .split('\t')
        .collect();
    if values.len() != 12 && values.len() != 13 {
        return Err("CUPTI summary status row is malformed".into());
    }
    let quality = CuptiQuality {
        register_result: values[0].parse()?,
        enable_result: values[1].parse()?,
        runtime_enable_result: values[2].parse()?,
        buffers_requested: values[3].parse()?,
        buffers_completed: values[4].parse()?,
        buffer_exhaustions: values[5].parse()?,
        records_seen: values[6].parse()?,
        target_records: values[7].parse()?,
        invalid_records: values[8].parse()?,
        dropped_records: values[9].parse()?,
        api_records: values[10].parse()?,
        api_drops: values[11].parse()?,
        driver_enable_result: if values.len() == 13 {
            values[12].parse()?
        } else {
            0
        },
    };
    let clock_line = |name: &str| -> Result<ClockPair, Box<dyn Error>> {
        let line = lines
            .iter()
            .find(|line| line.starts_with(name) && line.as_bytes().get(name.len()) == Some(&b'\t'))
            .ok_or_else(|| format!("CUPTI summary lacks {name}"))?;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 5 {
            return Err(format!("CUPTI clock row {name} is malformed").into());
        }
        Ok(ClockPair {
            cupti_ns: fields[1].parse()?,
            monotonic_ns: fields[2].parse()?,
            uncertainty_ns: fields[3].parse()?,
        })
    };
    Ok((
        quality,
        ClockMap {
            start: clock_line("CLOCK_MONOTONIC_START")?,
            end: clock_line("CLOCK_MONOTONIC_END")?,
        },
    ))
}

fn bounded_file(path: &Path, max_bytes: u64) -> Result<File, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > max_bytes {
        return Err(format!("invalid bounded input size {bytes} for {}", path.display()).into());
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_clock_normalization_handles_offset_drift() {
        let clock = ClockMap {
            start: ClockPair {
                cupti_ns: 100,
                monotonic_ns: 1_100,
                uncertainty_ns: 3,
            },
            end: ClockPair {
                cupti_ns: 200,
                monotonic_ns: 1_210,
                uncertainty_ns: 4,
            },
        };
        assert_eq!(clock.normalize(150).unwrap(), 1_155);
    }

    #[test]
    fn interval_union_separates_sum_busy_and_gaps() {
        assert_eq!(interval_union(&[(10, 20), (15, 30), (40, 45)]).unwrap(), 25);
    }

    #[test]
    fn qwen_kernel_families_are_explicit_and_conservative() {
        assert_eq!(
            classify_kernel("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2"),
            KernelFamily::LayerNorm
        );
        assert_eq!(
            classify_kernel("internal::gemvx::kernel"),
            KernelFamily::Gemm
        );
        assert_eq!(
            classify_kernel("flash_fwd_splitkv_kernel"),
            KernelFamily::Attention
        );
        assert_eq!(
            classify_kernel("reshape_and_cache_flash_kernel"),
            KernelFamily::KvCache
        );
        assert_eq!(
            classify_kernel("unknown_future_kernel"),
            KernelFamily::Other
        );
    }
}
