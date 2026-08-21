#[cfg(test)]
use gpu_observer_collector::join_b_trace::OwnershipSlice;
use gpu_observer_collector::join_b_trace::{load_semantic, SemanticTrace};
use std::{
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

const MAX_ACTIVITY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RUNTIME_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SUMMARY_BYTES: u64 = 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;

#[derive(Clone, Copy, Debug)]
struct KernelActivity {
    start_ns: u64,
    end_ns: u64,
    stream: u32,
    correlation: u32,
    graph_node_id: u64,
    graph_id: u32,
    grid: [u32; 3],
    block: [u32; 3],
}

#[derive(Clone, Copy, Debug)]
struct RuntimeActivity {
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
    runtime_records: u64,
    runtime_drops: u64,
    driver_enable_result: i32,
}

#[derive(Default)]
struct StepAggregate {
    intervals: Vec<(u64, u64)>,
    runtime_first_ns: u64,
    runtime_last_ns: u64,
    streams: Vec<u32>,
    graph_nodes: u64,
    ordinary_kernels: u64,
    padding_blocks: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = PathBuf::from(args.next().unwrap_or_default());
    let usage = || {
        format!(
            "usage: {} SEMANTIC.bin ACTIVITIES.tsv RUNTIME.tsv SUMMARY.tsv --expected-kernels N --decode-suffix [--require-reordering]",
            program.display()
        )
    };
    let semantic_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let activity_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let runtime_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let summary_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let mut expected_kernels = None;
    let mut decode_suffix = false;
    let mut require_reordering = false;
    while let Some(arg) = args.next() {
        if arg == "--expected-kernels" {
            expected_kernels = Some(
                args.next()
                    .ok_or_else(&usage)?
                    .to_str()
                    .ok_or("expected kernel count is not UTF-8")?
                    .parse::<usize>()?,
            );
        } else if arg == "--decode-suffix" {
            decode_suffix = true;
        } else if arg == "--require-reordering" {
            require_reordering = true;
        } else {
            return Err(usage().into());
        }
    }
    let expected_kernels = expected_kernels.ok_or_else(&usage)?;
    if expected_kernels == 0 || expected_kernels > 8192 || !decode_suffix {
        return Err("expected kernels must be in 1..=8192 and --decode-suffix is required".into());
    }

    let semantic = load_semantic(&semantic_path)?;
    let mut kernels = load_kernels(&activity_path)?;
    let mut runtimes = load_runtimes(&runtime_path)?;
    let (quality, clock) = load_summary(&summary_path)?;
    validate_inputs(&semantic, &kernels, &runtimes, &quality, &clock)?;

    kernels.sort_unstable_by_key(|kernel| (kernel.start_ns, kernel.end_ns));
    runtimes.sort_unstable_by_key(|runtime| runtime.correlation);
    if runtimes
        .windows(2)
        .any(|pair| pair[0].correlation == pair[1].correlation)
    {
        return Err("CUPTI runtime correlation IDs are not unique".into());
    }

    let packed_timeline = packed_timeline(&semantic)?;
    let last_prefill_packed_ns = semantic
        .steps
        .iter()
        .filter(|step| step.begin.prefill_tokens != 0)
        .filter_map(|step| step.packed_begin.map(|packed| packed.timestamp_ns))
        .max()
        .ok_or("semantic trace contains no prefill step")?;
    let mut eligible = Vec::new();
    eligible.try_reserve(semantic.steps.len())?;
    for (index, step) in semantic.steps.iter().enumerate() {
        let packed = step.packed_begin.ok_or("step is missing packed layout")?;
        if packed.timestamp_ns > last_prefill_packed_ns
            && step.begin.prefill_tokens == 0
            && step.begin.decode_tokens != 0
        {
            eligible.push(index);
        }
    }
    if eligible.is_empty() {
        return Err("decode suffix after the final prefill is empty".into());
    }

    let mut aggregates = zeroed_aggregates(semantic.steps.len())?;
    let mut unmatched_correlations = 0_u64;
    let mut unassigned_kernels = 0_u64;
    let mut kernels_outside_suffix = 0_u64;
    let mut runtime_pid_mismatches = 0_u64;
    let mut invalid_geometry = 0_u64;

    for kernel in &kernels {
        let runtime = runtimes
            .binary_search_by_key(&kernel.correlation, |runtime| runtime.correlation)
            .ok()
            .map(|index| runtimes[index]);
        let Some(runtime) = runtime else {
            unmatched_correlations += 1;
            continue;
        };
        let runtime_start = clock.normalize(runtime.start_ns)?;
        let runtime_end = clock.normalize(runtime.end_ns)?;
        let kernel_start = clock.normalize(kernel.start_ns)?;
        let kernel_end = clock.normalize(kernel.end_ns)?;
        if runtime.return_value != 0 || runtime_end < runtime_start || kernel_end <= kernel_start {
            return Err("CUPTI runtime or kernel interval is invalid".into());
        }
        let Some(step_index) =
            step_for_packed_timestamp(runtime_start, &semantic, &packed_timeline)
        else {
            unassigned_kernels += 1;
            continue;
        };
        if eligible.binary_search(&step_index).is_err() {
            kernels_outside_suffix += 1;
            continue;
        }
        let step = &semantic.steps[step_index];
        if runtime.process != step.begin.pid {
            runtime_pid_mismatches += 1;
        }
        if kernel.grid[1] != 1
            || kernel.grid[2] != 1
            || kernel.grid[0] < step.begin.scheduled_tokens
            || kernel.block.contains(&0)
        {
            invalid_geometry += 1;
            continue;
        }
        let aggregate = &mut aggregates[step_index];
        aggregate.intervals.try_reserve(1)?;
        aggregate.intervals.push((kernel_start, kernel_end));
        if aggregate.runtime_first_ns == 0 || runtime_start < aggregate.runtime_first_ns {
            aggregate.runtime_first_ns = runtime_start;
        }
        aggregate.runtime_last_ns = aggregate.runtime_last_ns.max(runtime_end);
        if !aggregate.streams.contains(&kernel.stream) {
            aggregate.streams.try_reserve(1)?;
            aggregate.streams.push(kernel.stream);
        }
        aggregate.graph_nodes += u64::from(kernel.graph_node_id != 0 || kernel.graph_id != 0);
        aggregate.ordinary_kernels += u64::from(kernel.graph_node_id == 0 && kernel.graph_id == 0);
        aggregate.padding_blocks += u64::from(kernel.grid[0] - step.begin.scheduled_tokens);
    }

    let mut selected_reordered_steps = 0_u64;
    let mut kernel_count_mismatches = 0_u64;
    let mut request_count_mismatches = 0_u64;
    let mut total_kernel_sum_ns = 0_u64;
    let mut total_busy_union_ns = 0_u64;
    let mut total_selected_kernels = 0_u64;
    let mut total_padding_blocks = 0_u64;
    let mut graph_kernels = 0_u64;
    let mut ordinary_kernels = 0_u64;

    for &step_index in &eligible {
        let step = &semantic.steps[step_index];
        let aggregate = &mut aggregates[step_index];
        if step.scheduler_order_mismatches != 0 {
            selected_reordered_steps += 1;
        }
        if aggregate.intervals.len() != expected_kernels {
            kernel_count_mismatches += 1;
        }
        aggregate
            .intervals
            .sort_unstable_by_key(|interval| interval.0);
        let kernel_sum_ns = aggregate
            .intervals
            .iter()
            .try_fold(0_u64, |sum, &(start, end)| {
                sum.checked_add(end - start)
                    .ok_or("kernel duration sum overflow")
            })?;
        let busy_union_ns = interval_union(&aggregate.intervals)?;
        let gpu_first = aggregate.intervals.first().map_or(0, |interval| interval.0);
        let gpu_last = aggregate
            .intervals
            .iter()
            .map(|interval| interval.1)
            .max()
            .unwrap_or(0);
        let packed_ns = step
            .packed_begin
            .ok_or("step is missing packed layout")?
            .timestamp_ns;
        let first_gpu_lag_ns = signed_delta(gpu_first, packed_ns);
        let gpu_tail_to_step_end_ns = signed_delta(step.end_timestamp_ns, gpu_last);
        let overlap_ns = kernel_sum_ns.saturating_sub(busy_union_ns);

        println!(
            "step={} packing_generation={} scheduled={} decode={} requests={} kernels={} graph_kernels={} ordinary_kernels={} streams={} runtime_submit_span_ns={} kernel_sum_ns={} busy_union_ns={} overlap_ns={} first_gpu_lag_ns={} gpu_tail_to_step_end_ns={} padding_blocks={} reordered_positions={}",
            step.begin.step_id,
            step.packed_begin.map_or(0, |packed| packed.sequence_id),
            step.begin.scheduled_tokens,
            step.begin.decode_tokens,
            step.owner_count,
            aggregate.intervals.len(),
            aggregate.graph_nodes,
            aggregate.ordinary_kernels,
            aggregate.streams.len(),
            aggregate.runtime_last_ns.saturating_sub(aggregate.runtime_first_ns),
            kernel_sum_ns,
            busy_union_ns,
            overlap_ns,
            first_gpu_lag_ns,
            gpu_tail_to_step_end_ns,
            aggregate.padding_blocks,
            step.scheduler_order_mismatches,
        );

        let owners = &semantic.owners[step.owner_start..step.owner_start + step.owner_count];
        for owner in owners {
            let inferred_blocks =
                u64::from(owner.scheduled_tokens) * u64::try_from(aggregate.intervals.len())?;
            let expected = u64::from(owner.row_end - owner.row_begin)
                * u64::try_from(aggregate.intervals.len())?;
            request_count_mismatches +=
                u64::from(!owner.authoritative || inferred_blocks != expected);
            println!(
                "  request=0x{:016x} phase={} rows=[{},{}) geometry_inferred_blocks={} evidence=authoritative_packed_rows_plus_CUPTI_launch_geometry",
                owner.request_id,
                owner.phase,
                owner.row_begin,
                owner.row_end,
                inferred_blocks,
            );
        }

        total_selected_kernels += u64::try_from(aggregate.intervals.len())?;
        total_kernel_sum_ns = total_kernel_sum_ns
            .checked_add(kernel_sum_ns)
            .ok_or("total kernel sum overflow")?;
        total_busy_union_ns = total_busy_union_ns
            .checked_add(busy_union_ns)
            .ok_or("total busy union overflow")?;
        total_padding_blocks += aggregate.padding_blocks;
        graph_kernels += aggregate.graph_nodes;
        ordinary_kernels += aggregate.ordinary_kernels;
    }

    let drift_ns = clock.end.offset_ns() - clock.start.offset_ns();
    let passed = unmatched_correlations == 0
        && unassigned_kernels == 0
        && runtime_pid_mismatches == 0
        && invalid_geometry == 0
        && kernel_count_mismatches == 0
        && request_count_mismatches == 0
        && (!require_reordering || selected_reordered_steps != 0);
    println!(
        "summary status={} semantic_records={} semantic_loss={} packed_steps={} max_steps_in_flight={} scheduler_order_mismatch_steps={} eligible_decode_steps={} selected_reordered_steps={} cupti_records_seen={} target_kernels={} selected_kernels={} kernels_outside_suffix={} graph_kernels={} ordinary_kernels={} runtime_records={} buffers_requested={} buffers_completed={} unmatched_correlations={} unassigned_kernels={} runtime_pid_mismatches={} invalid_geometry={} kernel_count_mismatches={} request_count_mismatches={} kernel_sum_ns={} busy_union_ns={} overlap_ns={} padding_blocks={} clock_start_uncertainty_ns={} clock_end_uncertainty_ns={} clock_offset_drift_ns={} assignment=runtime_correlation_then_packed_window timing=normalized_CUPTI_actual_intervals ownership=authoritative_packed_rows_plus_CUPTI_launch_geometry device_internal_events=absent",
        if passed { "PASS" } else { "FAIL" },
        semantic.records,
        semantic.loss_markers,
        semantic.packed_steps,
        semantic.max_steps_in_flight,
        semantic.scheduler_order_mismatch_steps,
        eligible.len(),
        selected_reordered_steps,
        quality.records_seen,
        kernels.len(),
        total_selected_kernels,
        kernels_outside_suffix,
        graph_kernels,
        ordinary_kernels,
        quality.runtime_records,
        quality.buffers_requested,
        quality.buffers_completed,
        unmatched_correlations,
        unassigned_kernels,
        runtime_pid_mismatches,
        invalid_geometry,
        kernel_count_mismatches,
        request_count_mismatches,
        total_kernel_sum_ns,
        total_busy_union_ns,
        total_kernel_sum_ns.saturating_sub(total_busy_union_ns),
        total_padding_blocks,
        clock.start.uncertainty_ns,
        clock.end.uncertainty_ns,
        drift_ns,
    );
    if !passed {
        return Err("CUPTI Join B quality gate failed".into());
    }
    Ok(())
}

fn validate_inputs(
    semantic: &SemanticTrace,
    kernels: &[KernelActivity],
    runtimes: &[RuntimeActivity],
    quality: &CuptiQuality,
    clock: &ClockMap,
) -> Result<(), Box<dyn Error>> {
    if semantic.steps.is_empty()
        || semantic.loss_markers != 0
        || semantic.packed_steps != semantic.steps.len() as u64
    {
        return Err("semantic packed-layout gate failed".into());
    }
    if kernels.is_empty() || runtimes.is_empty() {
        return Err("CUPTI kernel or runtime trace is empty".into());
    }
    if quality.register_result != 0
        || quality.enable_result != 0
        || quality.runtime_enable_result != 0
        || quality.driver_enable_result != 0
        || quality.buffer_exhaustions != 0
        || quality.invalid_records != 0
        || quality.dropped_records != 0
        || quality.runtime_drops != 0
        || quality.target_records != kernels.len() as u64
        || quality.runtime_records != runtimes.len() as u64
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
    timeline.try_reserve(semantic.steps.len())?;
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

#[cfg(test)]
fn owner_for_row(owners: &[OwnershipSlice], row: u32) -> Option<usize> {
    let position = owners.partition_point(|owner| owner.row_end <= row);
    owners
        .get(position)
        .filter(|owner| owner.authoritative && row >= owner.row_begin && row < owner.row_end)
        .map(|_| position)
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
            grid: [fields[9].parse()?, fields[10].parse()?, fields[11].parse()?],
            block: [
                fields[12].parse()?,
                fields[13].parse()?,
                fields[14].parse()?,
            ],
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

fn load_runtimes(path: &Path) -> Result<Vec<RuntimeActivity>, Box<dyn Error>> {
    let file = bounded_file(path, MAX_RUNTIME_BYTES)?;
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
            return Err("CUPTI runtime trace exceeds collector bound".into());
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 7 {
            return Err(format!("runtime row {} is malformed", line_number + 1).into());
        }
        let activity = RuntimeActivity {
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
            return Err(format!("runtime row {} is invalid", line_number + 1).into());
        }
        output.try_reserve(1)?;
        output.push(activity);
    }
    if !format || !header {
        return Err("CUPTI runtime trace lacks its versioned header".into());
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
        runtime_records: values[10].parse()?,
        runtime_drops: values[11].parse()?,
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
    fn interval_union_separates_sum_from_busy_time() {
        assert_eq!(interval_union(&[(10, 20), (15, 30), (40, 45)]).unwrap(), 25);
    }

    #[test]
    fn owner_mapping_uses_authoritative_half_open_rows() {
        let owners = [
            OwnershipSlice {
                request_id: 1,
                phase: 2,
                scheduled_tokens: 2,
                row_begin: 0,
                row_end: 2,
                authoritative: true,
            },
            OwnershipSlice {
                request_id: 2,
                phase: 2,
                scheduled_tokens: 1,
                row_begin: 2,
                row_end: 3,
                authoritative: true,
            },
        ];
        assert_eq!(owner_for_row(&owners, 0), Some(0));
        assert_eq!(owner_for_row(&owners, 2), Some(1));
        assert_eq!(owner_for_row(&owners, 3), None);
    }
}
