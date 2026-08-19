use gpu_observer_core::{SemanticRecordKind, SemanticWireRecord, SEMANTIC_RECORD_BYTES};
use std::{
    env,
    error::Error,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};
const MAX_INPUT_BYTES: u64 = 256 * 1024 * 1024;
fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let input_path = PathBuf::from(
        arguments
            .next()
            .ok_or_else(|| format!("usage: {} SEMANTIC.bin", PathBuf::from(&program).display()))?,
    );
    if arguments.next().is_some() {
        return Err("expected exactly one semantic trace".into());
    }
    let file = File::open(&input_path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_INPUT_BYTES || bytes % SEMANTIC_RECORD_BYTES as u64 != 0 {
        return Err(format!("invalid semantic trace length: {bytes}").into());
    }
    let mut input = BufReader::with_capacity(1024 * 1024, file);
    let mut raw = [0_u8; SEMANTIC_RECORD_BYTES];
    let mut records = 0_u64;
    let mut previous_sequence = None;
    while input.read_exact(&mut raw).is_ok() {
        let record = SemanticWireRecord::decode(&raw)
            .map_err(|error| format!("invalid record {records}: {error:?}"))?;
        if previous_sequence.is_some_and(|previous| record.sequence != previous + 1) {
            return Err(format!(
                "sequence gap before record {records}: previous={previous_sequence:?} current={}",
                record.sequence
            )
            .into());
        }
        previous_sequence = Some(record.sequence);
        print_record(record);
        records += 1;
    }
    eprintln!("summary records={records}");
    Ok(())
}
fn print_record(record: SemanticWireRecord) {
    match record.kind {
        SemanticRecordKind::ENGINE_STEP_BEGIN => println!(
            "begin ts={} seq={} step={} scheduled={} prefill={} decode={} queue={} active={} slices={} kv_permyriad={} flags=0x{:x}",
            record.timestamp_ns,
            record.sequence,
            record.step_id,
            record.scheduled_tokens,
            record.prefill_tokens,
            record.decode_tokens,
            record.queue_depth,
            record.active_requests,
            record.expected_slices,
            record.kv_cache_usage_permyriad,
            record.flags,
        ),
        SemanticRecordKind::STEP_REQUEST_SLICE => println!(
            "slice ts={} seq={} step={} request=0x{:016x} phase={} tokens={}",
            record.timestamp_ns,
            record.sequence,
            record.step_id,
            record.request_id,
            record.phase,
            record.scheduled_tokens,
        ),
        SemanticRecordKind::PACKED_LAYOUT_BEGIN => println!(
            "packed_begin ts={} seq={} step={} generation={} tokens={} slices={} flags=0x{:x}",
            record.timestamp_ns,
            record.sequence,
            record.step_id,
            record.sequence_id,
            record.scheduled_tokens,
            record.expected_slices,
            record.flags,
        ),
        SemanticRecordKind::PACKED_REQUEST_SLICE => println!(
            "packed_slice ts={} seq={} step={} generation={} request=0x{:016x} index={} rows=[{},{}) tokens={} phase={}",
            record.timestamp_ns,
            record.sequence,
            record.step_id,
            record.sequence_id,
            record.request_id,
            record.queue_depth,
            record.prefill_tokens,
            record.decode_tokens,
            record.scheduled_tokens,
            record.phase,
        ),
        SemanticRecordKind::PACKED_TOKEN_ROW => println!(
            "packed_token ts={} seq={} step={} generation={} request=0x{:016x} row={} position={} token_id={} phase={} flags=0x{:x}",
            record.timestamp_ns,
            record.sequence,
            record.step_id,
            record.sequence_id,
            record.request_id,
            record.prefill_tokens,
            record.decode_tokens,
            record.scheduled_tokens,
            record.phase,
            record.flags,
        ),
        SemanticRecordKind::ACCEPTED_OUTPUT_TOKEN => println!(
            "accepted_token ts={} seq={} step={} request=0x{:016x} output_position={} token_id={} flags=0x{:x}",
            record.timestamp_ns,
            record.sequence,
            record.step_id,
            record.request_id,
            record.sequence_id,
            record.scheduled_tokens,
            record.flags,
        ),
        SemanticRecordKind::ENGINE_STEP_END => println!(
            "end ts={} seq={} step={} status={} flags=0x{:x}",
            record.timestamp_ns,
            record.sequence,
            record.step_id,
            record.status,
            record.flags,
        ),
        _ => unreachable!(),
    }
}
