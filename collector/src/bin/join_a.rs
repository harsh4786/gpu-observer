use std::{
    env,
    error::Error,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use gpu_observer_core::{
    SemanticRecordFlags, SemanticRecordKind, SemanticWireRecord, SEMANTIC_RECORD_BYTES,
};
use gpu_observer_host_probe_common::{CudaLaunchEvent, LaunchFlags, CUDA_LAUNCH_EVENT_BYTES};

const MAX_SEMANTIC_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CUDA_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Default)]
struct Step {
    begin: SemanticWireRecord,
    end_timestamp_ns: u64,
    slice_start: usize,
    slice_count: usize,
    launches: u64,
    regular_launches: u64,
    extended_launches: u64,
    first_launch_ns: Option<u64>,
    last_launch_ns: Option<u64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let usage = || {
        format!(
            "usage: {} SEMANTIC.bin CUDA-LAUNCHES.bin [HOST-PID]",
            PathBuf::from(&program).display()
        )
    };
    let semantic_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let cuda_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let host_pid = arguments
        .next()
        .map(|value| value.to_string_lossy().parse::<u32>())
        .transpose()?;
    if arguments.next().is_some() {
        return Err(usage().into());
    }

    let (mut steps, slices, semantic_records, semantic_loss_markers) =
        load_semantic(&semantic_path)?;
    if steps.is_empty() {
        return Err("semantic trace contains no engine steps".into());
    }
    let semantic_pid = steps[0].begin.pid;
    let expected_cuda_pid = host_pid.unwrap_or(semantic_pid);

    let mut cuda = open_fixed(
        &cuda_path,
        CUDA_LAUNCH_EVENT_BYTES,
        MAX_CUDA_BYTES,
        "CUDA launch",
    )?;
    let mut raw = [0_u8; CUDA_LAUNCH_EVENT_BYTES];
    let mut step_cursor = 0_usize;
    let mut cuda_records = 0_u64;
    let mut cuda_loss_markers = 0_u64;
    let mut sequence_gaps = 0_u64;
    let mut pid_mismatches = 0_u64;
    let mut unassigned = 0_u64;
    let mut cpu_sequences: Vec<(u32, u64)> = Vec::new();
    let mut tids: Vec<u32> = Vec::new();

    while cuda.read_exact(&mut raw).is_ok() {
        let launch = CudaLaunchEvent::decode(&raw)
            .map_err(|error| format!("invalid CUDA launch record {cuda_records}: {error:?}"))?;
        cuda_records += 1;
        if launch.has_flag(LaunchFlags::DROPPED_BEFORE) {
            cuda_loss_markers += 1;
        }
        if launch.pid != expected_cuda_pid {
            pid_mismatches += 1;
        }
        if !tids.contains(&launch.tid) {
            tids.try_reserve(1)?;
            tids.push(launch.tid);
        }

        match cpu_sequences
            .iter_mut()
            .find(|(cpu_id, _)| *cpu_id == launch.cpu_id)
        {
            Some((_, previous)) => {
                if launch.sequence != previous.wrapping_add(1) {
                    sequence_gaps += 1;
                }
                *previous = launch.sequence;
            }
            None => {
                cpu_sequences.try_reserve(1)?;
                cpu_sequences.push((launch.cpu_id, launch.sequence));
            }
        }

        while step_cursor < steps.len() && launch.timestamp_ns > steps[step_cursor].end_timestamp_ns
        {
            step_cursor += 1;
        }
        if step_cursor == steps.len() || launch.timestamp_ns < steps[step_cursor].begin.timestamp_ns
        {
            unassigned += 1;
            continue;
        }

        let step = &mut steps[step_cursor];
        step.launches += 1;
        if launch.has_flag(LaunchFlags::EXTENDED_CONFIG) {
            step.extended_launches += 1;
        } else {
            step.regular_launches += 1;
        }
        step.first_launch_ns.get_or_insert(launch.timestamp_ns);
        step.last_launch_ns = Some(launch.timestamp_ns);
    }

    let assigned: u64 = steps.iter().map(|step| step.launches).sum();
    for step in &steps {
        let wall_ns = step
            .end_timestamp_ns
            .saturating_sub(step.begin.timestamp_ns);
        let submission_span_ns = match (step.first_launch_ns, step.last_launch_ns) {
            (Some(first), Some(last)) => last.saturating_sub(first),
            _ => 0,
        };
        println!(
            "step={} begin_ns={} end_ns={} wall_us={:.3} scheduled={} prefill={} decode={} queue={} active={} kv_permyriad={} launches={} regular={} extended={} submission_span_us={:.3}",
            step.begin.step_id,
            step.begin.timestamp_ns,
            step.end_timestamp_ns,
            wall_ns as f64 / 1_000.0,
            step.begin.scheduled_tokens,
            step.begin.prefill_tokens,
            step.begin.decode_tokens,
            step.begin.queue_depth,
            step.begin.active_requests,
            step.begin.kv_cache_usage_permyriad,
            step.launches,
            step.regular_launches,
            step.extended_launches,
            submission_span_ns as f64 / 1_000.0,
        );
        for slice in &slices[step.slice_start..step.slice_start + step.slice_count] {
            println!(
                "  request=0x{:016x} phase={} tokens={} sequence_id={}",
                slice.request_id, slice.phase, slice.scheduled_tokens, slice.sequence_id
            );
        }
    }

    println!(
        "summary semantic_records={} steps={} request_slices={} semantic_loss_markers={} cuda_records={} assigned={} unassigned={} cuda_loss_markers={} sequence_gaps={} pid_mismatches={} semantic_pid={} expected_cuda_pid={} pid_namespace_remap={} launch_tids={} gpu_duration=unavailable_without_cupti",
        semantic_records,
        steps.len(),
        slices.len(),
        semantic_loss_markers,
        cuda_records,
        assigned,
        unassigned,
        cuda_loss_markers,
        sequence_gaps,
        pid_mismatches,
        semantic_pid,
        expected_cuda_pid,
        semantic_pid != expected_cuda_pid,
        tids.len(),
    );

    if semantic_loss_markers != 0
        || cuda_loss_markers != 0
        || sequence_gaps != 0
        || pid_mismatches != 0
        || unassigned != 0
    {
        return Err("trace quality checks failed".into());
    }
    Ok(())
}

fn load_semantic(
    path: &Path,
) -> Result<(Vec<Step>, Vec<SemanticWireRecord>, u64, u64), Box<dyn Error>> {
    let mut input = open_fixed(path, SEMANTIC_RECORD_BYTES, MAX_SEMANTIC_BYTES, "semantic")?;
    let record_count = input.get_ref().metadata()?.len() as usize / SEMANTIC_RECORD_BYTES;
    let mut steps = Vec::new();
    let mut slices = Vec::new();
    steps.try_reserve(record_count / 3 + 1)?;
    slices.try_reserve(record_count / 3 + 1)?;

    let mut raw = [0_u8; SEMANTIC_RECORD_BYTES];
    let mut records = 0_u64;
    let mut loss_markers = 0_u64;
    let mut previous_sequence = None;
    while input.read_exact(&mut raw).is_ok() {
        let record = SemanticWireRecord::decode(&raw)
            .map_err(|error| format!("invalid semantic record {records}: {error:?}"))?;
        if let Some(previous) = previous_sequence {
            if record.sequence != previous + 1 {
                return Err(format!(
                    "semantic sequence gap: previous={previous} current={}",
                    record.sequence
                )
                .into());
            }
        }
        previous_sequence = Some(record.sequence);
        records += 1;
        if record.flags & SemanticRecordFlags::DROPPED_BEFORE != 0 {
            loss_markers += 1;
        }

        match record.kind {
            SemanticRecordKind::ENGINE_STEP_BEGIN => {
                if steps
                    .last()
                    .is_some_and(|step: &Step| step.end_timestamp_ns == 0)
                {
                    return Err("overlapping or unterminated semantic step".into());
                }
                steps.try_reserve(1)?;
                steps.push(Step {
                    begin: record,
                    slice_start: slices.len(),
                    ..Step::default()
                });
            }
            SemanticRecordKind::STEP_REQUEST_SLICE => {
                let step = steps.last_mut().ok_or("slice before step begin")?;
                if step.begin.step_id != record.step_id || step.end_timestamp_ns != 0 {
                    return Err("slice is outside its dense step range".into());
                }
                slices.try_reserve(1)?;
                slices.push(record);
                step.slice_count += 1;
            }
            SemanticRecordKind::PACKED_LAYOUT_BEGIN
            | SemanticRecordKind::PACKED_REQUEST_SLICE
            | SemanticRecordKind::PACKED_TOKEN_ROW
            | SemanticRecordKind::ACCEPTED_OUTPUT_TOKEN => {}
            SemanticRecordKind::ENGINE_STEP_END => {
                let step = steps.last_mut().ok_or("step end before begin")?;
                if step.begin.step_id != record.step_id || step.end_timestamp_ns != 0 {
                    return Err("mismatched or duplicate step end".into());
                }
                if step.slice_count != step.begin.expected_slices as usize {
                    return Err("request-slice count does not match step header".into());
                }
                let token_sum: u32 = slices[step.slice_start..step.slice_start + step.slice_count]
                    .iter()
                    .map(|slice| slice.scheduled_tokens)
                    .sum();
                if token_sum != step.begin.scheduled_tokens {
                    return Err("request-slice token sum does not match step header".into());
                }
                if record.status != 0 || record.timestamp_ns < step.begin.timestamp_ns {
                    return Err("failed step or negative semantic interval".into());
                }
                step.end_timestamp_ns = record.timestamp_ns;
            }
            _ => unreachable!(),
        }
    }
    if steps.iter().any(|step| step.end_timestamp_ns == 0) {
        return Err("semantic trace ends with an unterminated step".into());
    }
    Ok((steps, slices, records, loss_markers))
}

fn open_fixed(
    path: &Path,
    record_bytes: usize,
    max_bytes: u64,
    label: &str,
) -> Result<BufReader<File>, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes > max_bytes {
        return Err(format!("{label} trace exceeds bounded input budget").into());
    }
    if bytes == 0 || bytes % record_bytes as u64 != 0 {
        return Err(format!("{label} trace has an invalid byte length: {bytes}").into());
    }
    Ok(BufReader::with_capacity(1024 * 1024, file))
}
