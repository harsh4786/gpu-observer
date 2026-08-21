use std::env;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use gpu_observer_collector::semantic::SemanticRingReader;
use gpu_observer_core::{SemanticRecordKind, SemanticWireRecord};

const DEFAULT_DURATION_SECONDS: u64 = 15;
const IDLE_POLL_MICROS: u64 = 200;
const SAMPLE_LIMIT: u64 = 12;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let usage = || {
        format!(
            "usage: {} SHARED_RING OUTPUT.bin [duration-seconds]",
            PathBuf::from(&program).display()
        )
    };
    let shared_ring = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let output = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let duration = match arguments.next() {
        Some(value) => Duration::from_secs(value.to_string_lossy().parse()?),
        None => Duration::from_secs(DEFAULT_DURATION_SECONDS),
    };
    if arguments.next().is_some() {
        return Err(usage().into());
    }

    let mut reader = SemanticRingReader::open(&shared_ring)?;
    let output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)?;
    let mut output = BufWriter::with_capacity(1024 * 1024, output);
    let deadline = Instant::now() + duration;
    let mut records = 0_u64;
    let mut begins = 0_u64;
    let mut slices = 0_u64;
    let mut packed_begins = 0_u64;
    let mut packed_slices = 0_u64;
    let mut packed_tokens = 0_u64;
    let mut accepted_tokens = 0_u64;
    let mut lifecycle = 0_u64;
    let mut ends = 0_u64;

    loop {
        let mut drained = false;
        while let Some(record) = reader.try_next()? {
            drained = true;
            output.write_all(record.as_bytes())?;
            records += 1;
            match record.kind {
                SemanticRecordKind::ENGINE_STEP_BEGIN => begins += 1,
                SemanticRecordKind::STEP_REQUEST_SLICE => slices += 1,
                SemanticRecordKind::PACKED_LAYOUT_BEGIN => packed_begins += 1,
                SemanticRecordKind::PACKED_REQUEST_SLICE => packed_slices += 1,
                SemanticRecordKind::PACKED_TOKEN_ROW => packed_tokens += 1,
                SemanticRecordKind::ACCEPTED_OUTPUT_TOKEN => accepted_tokens += 1,
                SemanticRecordKind::ENGINE_STEP_END => ends += 1,
                _ => lifecycle += 1,
            }
            if records <= SAMPLE_LIMIT {
                print_record(record);
            }
        }

        if Instant::now() >= deadline {
            break;
        }
        if !drained {
            thread::sleep(Duration::from_micros(IDLE_POLL_MICROS));
        }
    }

    output.flush()?;
    output.get_ref().sync_all()?;
    println!(
        "summary records={} begins={} slices={} packed_begins={} packed_slices={} packed_tokens={} accepted_tokens={} lifecycle={} ends={} producer_dropped={}",
        records,
        begins,
        slices,
        packed_begins,
        packed_slices,
        packed_tokens,
        accepted_tokens,
        lifecycle,
        ends,
        reader.dropped_records(),
    );
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
        _ => println!(
            "lifecycle ts={} seq={} kind={} request=0x{:016x} peer=0x{:016x} position={} token_id={} queue={} status={} flags=0x{:x}",
            record.timestamp_ns,
            record.sequence,
            record.kind,
            record.request_id,
            record.sequence_id,
            record.prefill_tokens,
            record.scheduled_tokens,
            record.queue_depth,
            record.status,
            record.flags,
        ),
    }
}
