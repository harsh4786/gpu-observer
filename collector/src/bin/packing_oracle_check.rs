use gpu_observer_core::{
    SemanticRecordFlags, SemanticRecordKind, SemanticWireRecord, SEMANTIC_RECORD_BYTES,
};
use std::{
    env,
    error::Error,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

const ORACLE_MAGIC: u32 = u32::from_le_bytes(*b"GPOP");
const ORACLE_VERSION: u16 = 1;
const ORACLE_BEGIN: u16 = 1;
const ORACLE_SLICE: u16 = 2;
const ORACLE_COMPLETE_FLAGS: u32 = 0b111;
const ORACLE_RECORD_BYTES: usize = 64;
const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Default)]
struct Step {
    begin: SemanticWireRecord,
    end_timestamp_ns: u64,
    slices: Vec<SemanticWireRecord>,
}

#[derive(Clone, Copy)]
struct OracleSlice {
    request_id: u64,
    row_begin: u32,
    row_end: u32,
    scheduled_tokens: u32,
    packed_index: u32,
}

struct OracleObservation {
    timestamp_ns: u64,
    request_count: u32,
    total_tokens: u32,
    flags: u32,
    pid: u32,
    slices: Vec<OracleSlice>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let usage = || {
        format!(
            "usage: {} SEMANTIC.bin PACKING-ORACLE.bin",
            PathBuf::from(&program).display()
        )
    };
    let semantic_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let oracle_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    if arguments.next().is_some() {
        return Err(usage().into());
    }

    let (steps, semantic_records, semantic_loss) = load_semantic(&semantic_path)?;
    let (observations, oracle_records) = load_oracle(&oracle_path)?;
    if steps.is_empty() || observations.is_empty() {
        return Err("packing oracle requires non-empty semantic and oracle streams".into());
    }

    let mut step_observation = vec![None; steps.len()];
    let mut assigned = 0_u64;
    let mut unassigned = 0_u64;
    let mut duplicate_steps = 0_u64;
    let mut flag_failures = 0_u64;
    let mut range_mismatches = 0_u64;
    let mut mixed_steps = 0_u64;

    for (observation_index, observation) in observations.iter().enumerate() {
        let Some(step_index) = step_for_timestamp(observation.timestamp_ns, &steps) else {
            unassigned += 1;
            continue;
        };
        assigned += 1;
        if step_observation[step_index]
            .replace(observation_index)
            .is_some()
        {
            duplicate_steps += 1;
        }
        if observation.flags != ORACLE_COMPLETE_FLAGS {
            flag_failures += 1;
        }

        let step = &steps[step_index];
        if step.slices.len() > 1 {
            mixed_steps += 1;
        }
        let mut mismatches = 0_u64;
        if observation.request_count as usize != step.slices.len()
            || observation.total_tokens != step.begin.scheduled_tokens
            || observation.slices.len() != step.slices.len()
        {
            mismatches += 1;
        }

        let mut predicted_row_begin = 0_u32;
        for (packed_index, (predicted, actual)) in step
            .slices
            .iter()
            .zip(observation.slices.iter())
            .enumerate()
        {
            let predicted_row_end = predicted_row_begin.saturating_add(predicted.scheduled_tokens);
            if actual.request_id != predicted.request_id
                || actual.scheduled_tokens != predicted.scheduled_tokens
                || actual.row_begin != predicted_row_begin
                || actual.row_end != predicted_row_end
                || actual.packed_index as usize != packed_index
            {
                mismatches += 1;
            }
            predicted_row_begin = predicted_row_end;
        }
        range_mismatches += mismatches;

        println!(
            "step={} scheduled={} slices={} oracle_timestamp_ns={} oracle_pid={} packing={}",
            step.begin.step_id,
            step.begin.scheduled_tokens,
            step.slices.len(),
            observation.timestamp_ns,
            observation.pid,
            if mismatches == 0 && observation.flags == ORACLE_COMPLETE_FLAGS {
                "matched"
            } else {
                "mismatch"
            },
        );
        if step.slices.len() > 1 {
            for actual in &observation.slices {
                println!(
                    "  packed_request=0x{:016x} packed_index={} row_range=[{},{}) tokens={}",
                    actual.request_id,
                    actual.packed_index,
                    actual.row_begin,
                    actual.row_end,
                    actual.scheduled_tokens,
                );
            }
        }
    }

    let missing_steps = step_observation
        .iter()
        .filter(|observation| observation.is_none())
        .count() as u64;
    let passed = semantic_loss == 0
        && assigned == steps.len() as u64
        && missing_steps == 0
        && duplicate_steps == 0
        && flag_failures == 0
        && range_mismatches == 0
        && mixed_steps != 0;
    let ownership = if passed {
        "oracle_validated"
    } else {
        "oracle_rejected"
    };
    println!(
        "summary semantic_records={} semantic_loss={} steps={} oracle_records={} oracle_observations={} assigned_observations={} unassigned_observations={} missing_steps={} duplicate_steps={} flag_failures={} range_mismatches={} mixed_steps={} clock=clock_monotonic ownership={}",
        semantic_records,
        semantic_loss,
        steps.len(),
        oracle_records,
        observations.len(),
        assigned,
        unassigned,
        missing_steps,
        duplicate_steps,
        flag_failures,
        range_mismatches,
        mixed_steps,
        ownership,
    );

    if !passed {
        return Err("packing ownership oracle gate failed".into());
    }
    Ok(())
}

fn step_for_timestamp(timestamp_ns: u64, steps: &[Step]) -> Option<usize> {
    steps.iter().position(|step| {
        timestamp_ns >= step.begin.timestamp_ns && timestamp_ns <= step.end_timestamp_ns
    })
}

fn load_oracle(path: &Path) -> Result<(Vec<OracleObservation>, u64), Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_INPUT_BYTES || bytes % ORACLE_RECORD_BYTES as u64 != 0 {
        return Err(format!("invalid packing-oracle size: {bytes}").into());
    }
    let record_count = bytes as usize / ORACLE_RECORD_BYTES;
    let mut input = BufReader::with_capacity(1024 * 1024, file);
    let mut records = Vec::new();
    records.try_reserve(record_count)?;
    for record_index in 0..record_count {
        let mut raw = [0_u8; ORACLE_RECORD_BYTES];
        input.read_exact(&mut raw)?;
        if le_u32(&raw, 0) != ORACLE_MAGIC || le_u16(&raw, 4) != ORACLE_VERSION {
            return Err(format!("invalid packing-oracle record {record_index}").into());
        }
        records.push(raw);
    }

    let mut observations = Vec::new();
    observations.try_reserve(record_count / 2 + 1)?;
    let mut cursor = 0_usize;
    while cursor < records.len() {
        let begin = &records[cursor];
        if le_u16(begin, 6) != ORACLE_BEGIN {
            return Err(format!("packing slice without begin at record {cursor}").into());
        }
        let timestamp_ns = le_u64(begin, 8);
        let request_count = le_u32(begin, 48);
        let total_tokens = le_u32(begin, 52);
        let flags = le_u32(begin, 56);
        let pid = le_u32(begin, 60);
        if request_count == 0
            || request_count > 65_536
            || le_u64(begin, 16) != 0
            || le_u64(begin, 24) != 0
            || le_u32(begin, 32) != 0
            || le_u32(begin, 36) != total_tokens
            || le_u32(begin, 40) != total_tokens
            || le_u32(begin, 44) != 0
        {
            return Err(format!("invalid packing begin at record {cursor}").into());
        }
        let next = cursor
            .checked_add(1)
            .and_then(|value| value.checked_add(request_count as usize))
            .ok_or("packing-oracle record count overflow")?;
        if next > records.len() {
            return Err("truncated packing-oracle observation".into());
        }

        let mut slices = Vec::new();
        slices.try_reserve_exact(request_count as usize)?;
        let mut expected_row_begin = 0_u32;
        for packed_index in 0..request_count {
            let raw = &records[cursor + 1 + packed_index as usize];
            let row_begin = le_u32(raw, 32);
            let row_end = le_u32(raw, 36);
            let scheduled_tokens = le_u32(raw, 40);
            let actual_index = le_u32(raw, 44);
            if le_u16(raw, 6) != ORACLE_SLICE
                || le_u64(raw, 8) != timestamp_ns
                || le_u64(raw, 16) != 0
                || le_u32(raw, 48) != request_count
                || le_u32(raw, 52) != total_tokens
                || le_u32(raw, 56) != 0
                || le_u32(raw, 60) != pid
                || actual_index != packed_index
                || row_begin != expected_row_begin
                || row_end < row_begin
                || row_end - row_begin != scheduled_tokens
                || scheduled_tokens == 0
            {
                return Err(format!(
                    "invalid packing slice at record {}",
                    cursor + 1 + packed_index as usize
                )
                .into());
            }
            slices.push(OracleSlice {
                request_id: le_u64(raw, 24),
                row_begin,
                row_end,
                scheduled_tokens,
                packed_index: actual_index,
            });
            expected_row_begin = row_end;
        }
        if expected_row_begin != total_tokens {
            return Err("packing-oracle ranges do not cover total tokens".into());
        }
        observations.push(OracleObservation {
            timestamp_ns,
            request_count,
            total_tokens,
            flags,
            pid,
            slices,
        });
        cursor = next;
    }
    Ok((observations, record_count as u64))
}

fn load_semantic(path: &Path) -> Result<(Vec<Step>, u64, u64), Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_INPUT_BYTES || bytes % SEMANTIC_RECORD_BYTES as u64 != 0 {
        return Err(format!("invalid semantic trace size: {bytes}").into());
    }
    let record_count = bytes as usize / SEMANTIC_RECORD_BYTES;
    let mut input = BufReader::with_capacity(1024 * 1024, file);
    let mut steps: Vec<Step> = Vec::new();
    steps.try_reserve(record_count / 3 + 1)?;
    let mut previous_sequence = None;
    let mut loss_markers = 0_u64;

    for record_index in 0..record_count {
        let mut raw = [0_u8; SEMANTIC_RECORD_BYTES];
        input.read_exact(&mut raw)?;
        let record = SemanticWireRecord::decode(&raw)
            .map_err(|error| format!("invalid semantic record {record_index}: {error:?}"))?;
        if previous_sequence.is_some_and(|previous| record.sequence != previous + 1) {
            return Err("semantic sequence gap".into());
        }
        previous_sequence = Some(record.sequence);
        if record.flags & SemanticRecordFlags::DROPPED_BEFORE != 0 {
            loss_markers += 1;
        }

        match record.kind {
            SemanticRecordKind::ENGINE_STEP_BEGIN => {
                if steps.last().is_some_and(|step| step.end_timestamp_ns == 0) {
                    return Err("overlapping semantic steps".into());
                }
                steps.push(Step {
                    begin: record,
                    ..Step::default()
                });
            }
            SemanticRecordKind::STEP_REQUEST_SLICE => {
                let step = steps.last_mut().ok_or("slice before step begin")?;
                if step.begin.step_id != record.step_id || step.end_timestamp_ns != 0 {
                    return Err("semantic slice outside its step".into());
                }
                step.slices.try_reserve(1)?;
                step.slices.push(record);
            }
            SemanticRecordKind::PACKED_LAYOUT_BEGIN
            | SemanticRecordKind::PACKED_REQUEST_SLICE
            | SemanticRecordKind::PACKED_TOKEN_ROW
            | SemanticRecordKind::ACCEPTED_OUTPUT_TOKEN => {}
            SemanticRecordKind::ENGINE_STEP_END => {
                let step = steps.last_mut().ok_or("step end before begin")?;
                if step.begin.step_id != record.step_id || step.end_timestamp_ns != 0 {
                    return Err("mismatched semantic step end".into());
                }
                let scheduled_tokens: u32 =
                    step.slices.iter().map(|slice| slice.scheduled_tokens).sum();
                if step.slices.len() != step.begin.expected_slices as usize
                    || scheduled_tokens != step.begin.scheduled_tokens
                    || record.status != 0
                    || record.timestamp_ns < step.begin.timestamp_ns
                {
                    return Err("semantic step integrity failure".into());
                }
                step.end_timestamp_ns = record.timestamp_ns;
            }
            _ => unreachable!(),
        }
    }
    if steps.iter().any(|step| step.end_timestamp_ns == 0) {
        return Err("unterminated semantic step".into());
    }
    Ok((steps, record_count as u64, loss_markers))
}

fn le_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
