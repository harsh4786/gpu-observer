//! Cold, bounded EXP-0022 lifecycle join.
//!
//! This is offline analysis, so `std` is appropriate. Every file and vector is
//! bounded; the inference hot path remains the fixed-record `no_std` core.

use gpu_observer_core::{
    SemanticRecordFlags, SemanticRecordKind, SemanticWireRecord, SEMANTIC_RECORD_BYTES,
};
use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const MAX_TRACE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_STEP_BYTES: u64 = 8 * 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;

#[derive(Clone, Copy)]
struct StepTiming {
    step_id: u64,
    begin_ns: u64,
    first_gpu_ns: u64,
    last_gpu_ns: u64,
    end_ns: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = PathBuf::from(args.next().unwrap_or_default());
    let usage = || {
        format!(
            "usage: {} ENGINE.bin FRONTEND.bin CLIENT.bin STEPS.tsv TIMELINE.tsv TOKENS.tsv",
            program.display()
        )
    };
    let engine_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let frontend_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let client_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let steps_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let timeline_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let tokens_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    if args.next().is_some() {
        return Err(usage().into());
    }

    let engine = load_records(&engine_path)?;
    let frontend = load_records(&frontend_path)?;
    let client = load_records(&client_path)?;
    let steps = load_steps(&steps_path)?;

    let client_send = exactly_one(&client, SemanticRecordKind::CLIENT_REQUEST_SENT, None)?;
    let request_hash = client_send.request_id;
    let frontend_receive = exactly_one(
        &frontend,
        SemanticRecordKind::FRONTEND_REQUEST_RECEIVED,
        Some(request_hash),
    )?;
    let admission = exactly_one_peer(
        &engine,
        SemanticRecordKind::ENGINE_REQUEST_ADMITTED,
        request_hash,
    )?;
    let engine_request_hash = admission.request_id;
    let first_schedule = engine
        .iter()
        .filter(|record| {
            record.kind == SemanticRecordKind::STEP_REQUEST_SLICE
                && record.request_id == engine_request_hash
        })
        .min_by_key(|record| record.timestamp_ns)
        .copied()
        .ok_or("request never appeared in a scheduler step")?;

    let mut accepted = selected(
        &engine,
        SemanticRecordKind::ACCEPTED_OUTPUT_TOKEN,
        engine_request_hash,
    )?;
    let mut emitted = selected(
        &frontend,
        SemanticRecordKind::FRONTEND_TOKEN_EMITTED,
        request_hash,
    )?;
    let mut received = selected(
        &client,
        SemanticRecordKind::CLIENT_TOKEN_RECEIVED,
        request_hash,
    )?;
    accepted.sort_unstable_by_key(|record| record.sequence_id);
    emitted.sort_unstable_by_key(|record| record.prefill_tokens);
    received.sort_unstable_by_key(|record| record.prefill_tokens);
    if accepted.is_empty() || accepted.len() != emitted.len() || accepted.len() != received.len() {
        return Err(format!(
            "token count mismatch: accepted={} emitted={} received={}",
            accepted.len(),
            emitted.len(),
            received.len()
        )
        .into());
    }

    let frontend_complete = exactly_one(
        &frontend,
        SemanticRecordKind::FRONTEND_REQUEST_COMPLETED,
        Some(request_hash),
    )?;
    let client_complete = exactly_one(
        &client,
        SemanticRecordKind::CLIENT_REQUEST_COMPLETED,
        Some(request_hash),
    )?;
    if frontend_complete.status != 0 || client_complete.status != 0 {
        return Err("frontend or client completion status is nonzero".into());
    }

    require_order(
        "client_send->frontend_receive",
        client_send.timestamp_ns,
        frontend_receive.timestamp_ns,
    )?;
    require_order(
        "frontend_receive->engine_admission",
        frontend_receive.timestamp_ns,
        admission.timestamp_ns,
    )?;
    require_order(
        "engine_admission->first_schedule",
        admission.timestamp_ns,
        first_schedule.timestamp_ns,
    )?;

    let first_step = steps
        .get(&accepted[0].step_id)
        .copied()
        .ok_or("first accepted token references a step without GPU timing")?;
    require_order(
        "first_schedule->first_gpu",
        first_schedule.timestamp_ns,
        first_step.first_gpu_ns,
    )?;

    let token_file = File::create(&tokens_path)?;
    let mut token_output = BufWriter::new(token_file);
    writeln!(
        token_output,
        "position\ttoken_id\tstep_id\tstep_begin_ns\tfirst_gpu_ns\tlast_gpu_ns\tengine_accept_ns\tfrontend_emit_ns\tclient_receive_ns\tgpu_to_accept_ns\taccept_to_frontend_ns\tfrontend_to_client_ns\tclient_itl_ns"
    )?;
    let mut prior_client_ns = 0_u64;
    for index in 0..accepted.len() {
        let accepted_token = accepted[index];
        let emitted_token = emitted[index];
        let received_token = received[index];
        let position = u32::try_from(index)?;
        let accepted_position = u32::try_from(accepted_token.sequence_id)?;
        if accepted_position != position
            || emitted_token.prefill_tokens != position
            || received_token.prefill_tokens != position
            || accepted_token.scheduled_tokens != emitted_token.scheduled_tokens
            || accepted_token.scheduled_tokens != received_token.scheduled_tokens
        {
            return Err(format!("token identity mismatch at position {index}").into());
        }
        let step = steps
            .get(&accepted_token.step_id)
            .copied()
            .ok_or("accepted token references a step without GPU timing")?;
        require_order("step_begin->first_gpu", step.begin_ns, step.first_gpu_ns)?;
        require_order("first_gpu->last_gpu", step.first_gpu_ns, step.last_gpu_ns)?;
        require_order("last_gpu->step_end", step.last_gpu_ns, step.end_ns)?;
        require_order(
            "last_gpu->engine_accept",
            step.last_gpu_ns,
            accepted_token.timestamp_ns,
        )?;
        require_order(
            "engine_accept->frontend_emit",
            accepted_token.timestamp_ns,
            emitted_token.timestamp_ns,
        )?;
        require_order(
            "frontend_emit->client_receive",
            emitted_token.timestamp_ns,
            received_token.timestamp_ns,
        )?;
        let itl = if prior_client_ns == 0 {
            0
        } else {
            received_token.timestamp_ns - prior_client_ns
        };
        writeln!(
            token_output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            position,
            accepted_token.scheduled_tokens,
            accepted_token.step_id,
            step.begin_ns,
            step.first_gpu_ns,
            step.last_gpu_ns,
            accepted_token.timestamp_ns,
            emitted_token.timestamp_ns,
            received_token.timestamp_ns,
            accepted_token.timestamp_ns - step.last_gpu_ns,
            emitted_token.timestamp_ns - accepted_token.timestamp_ns,
            received_token.timestamp_ns - emitted_token.timestamp_ns,
            itl,
        )?;
        prior_client_ns = received_token.timestamp_ns;
    }
    token_output.flush()?;

    require_order(
        "frontend_complete->client_complete",
        frontend_complete.timestamp_ns,
        client_complete.timestamp_ns,
    )?;
    require_order(
        "last_client_token->client_complete",
        received.last().unwrap().timestamp_ns,
        client_complete.timestamp_ns,
    )?;

    let first_accept = accepted[0].timestamp_ns;
    let first_emit = emitted[0].timestamp_ns;
    let first_receive = received[0].timestamp_ns;
    let timeline_file = File::create(&timeline_path)?;
    let mut timeline = BufWriter::new(timeline_file);
    writeln!(
        timeline,
        "boundary\ttimestamp_ns\tdelta_from_previous_ns\tobserver\tendpoint"
    )?;
    let boundaries = [
        (
            "client_send_begin",
            client_send.timestamp_ns,
            "rust_client",
            "before socket write",
        ),
        (
            "frontend_handler_entry",
            frontend_receive.timestamp_ns,
            "vllm_frontend",
            "create_chat_completion entry after HTTP decode",
        ),
        (
            "engine_scheduler_admission",
            admission.timestamp_ns,
            "engine_core",
            "after Scheduler.add_request",
        ),
        (
            "first_scheduler_selection",
            first_schedule.timestamp_ns,
            "engine_core",
            "engine_step_begin",
        ),
        (
            "first_gpu_kernel_start",
            first_step.first_gpu_ns,
            "cupti",
            "actual kernel start",
        ),
        (
            "first_token_step_gpu_end",
            first_step.last_gpu_ns,
            "cupti",
            "actual final kernel end",
        ),
        (
            "engine_accepts_first_token",
            first_accept,
            "engine_core",
            "Scheduler.update_from_output returned token",
        ),
        (
            "frontend_yields_first_token",
            first_emit,
            "vllm_frontend",
            "immediately before SSE generator yield",
        ),
        (
            "client_receives_first_token",
            first_receive,
            "rust_client",
            "HTTP chunk read completion",
        ),
    ];
    let mut previous = boundaries[0].1;
    for (name, timestamp, observer, endpoint) in boundaries {
        writeln!(
            timeline,
            "{}\t{}\t{}\t{}\t{}",
            name,
            timestamp,
            timestamp.saturating_sub(previous),
            observer,
            endpoint,
        )?;
        previous = timestamp;
    }
    timeline.flush()?;

    let ttft_ns = first_receive - client_send.timestamp_ns;
    println!(
        "summary status=PASS request=0x{request_hash:016x} engine_request=0x{engine_request_hash:016x} tokens={} queue_depth_at_admission={} first_step={} client_to_frontend_ns={} frontend_to_admission_ns={} admission_to_schedule_ns={} schedule_to_gpu_ns={} gpu_span_ns={} gpu_to_accept_ns={} accept_to_frontend_ns={} frontend_to_client_ns={} ttft_ns={} engine_records={} frontend_records={} client_records={}",
        accepted.len(),
        admission.queue_depth,
        first_step.step_id,
        frontend_receive.timestamp_ns - client_send.timestamp_ns,
        admission.timestamp_ns - frontend_receive.timestamp_ns,
        first_schedule.timestamp_ns - admission.timestamp_ns,
        first_step.first_gpu_ns - first_schedule.timestamp_ns,
        first_step.last_gpu_ns - first_step.first_gpu_ns,
        first_accept - first_step.last_gpu_ns,
        first_emit - first_accept,
        first_receive - first_emit,
        ttft_ns,
        engine.len(),
        frontend.len(),
        client.len(),
    );
    Ok(())
}

fn load_records(path: &Path) -> Result<Vec<SemanticWireRecord>, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_TRACE_BYTES || bytes % SEMANTIC_RECORD_BYTES as u64 != 0 {
        return Err(format!("invalid lifecycle trace size {bytes}: {}", path.display()).into());
    }
    let count = usize::try_from(bytes / SEMANTIC_RECORD_BYTES as u64)?;
    if count > MAX_RECORDS {
        return Err("lifecycle record bound exceeded".into());
    }
    let mut records = Vec::new();
    records.try_reserve_exact(count)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut raw = [0_u8; SEMANTIC_RECORD_BYTES];
    let mut first_sequence = None;
    for index in 0..count {
        reader.read_exact(&mut raw)?;
        let record = SemanticWireRecord::decode(&raw)
            .map_err(|error| format!("invalid record {index} in {}: {error:?}", path.display()))?;
        let base = *first_sequence.get_or_insert(record.sequence);
        if record.sequence != base.wrapping_add(index as u64) {
            return Err(format!("sequence gap at record {index}: {}", path.display()).into());
        }
        if record.flags & SemanticRecordFlags::DROPPED_BEFORE != 0 {
            return Err(format!("loss marker in {}", path.display()).into());
        }
        records.push(record);
    }
    Ok(records)
}

fn load_steps(path: &Path) -> Result<BTreeMap<u64, StepTiming>, Box<dyn Error>> {
    let file = File::open(path)?;
    if file.metadata()?.len() == 0 || file.metadata()?.len() > MAX_STEP_BYTES {
        return Err("invalid full-step table size".into());
    }
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().ok_or("step table is empty")??;
    let names: Vec<&str> = header.split('\t').collect();
    let index = |name: &str| {
        names
            .iter()
            .position(|value| *value == name)
            .ok_or_else(|| format!("step table lacks {name}"))
    };
    let id_i = index("step_id")?;
    let begin_i = index("step_begin_ns")?;
    let first_gpu_i = index("first_gpu_start_ns")?;
    let last_gpu_i = index("last_gpu_end_ns")?;
    let end_i = index("step_end_ns")?;
    let max_index = [id_i, begin_i, first_gpu_i, last_gpu_i, end_i]
        .into_iter()
        .max()
        .unwrap();
    let mut output = BTreeMap::new();
    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() <= max_index {
            return Err("truncated full-step row".into());
        }
        let step = StepTiming {
            step_id: fields[id_i].parse()?,
            begin_ns: fields[begin_i].parse()?,
            first_gpu_ns: fields[first_gpu_i].parse()?,
            last_gpu_ns: fields[last_gpu_i].parse()?,
            end_ns: fields[end_i].parse()?,
        };
        if output.insert(step.step_id, step).is_some() {
            return Err("duplicate full-step ID".into());
        }
    }
    if output.is_empty() {
        return Err("full-step table has no rows".into());
    }
    Ok(output)
}

fn exactly_one(
    records: &[SemanticWireRecord],
    kind: u8,
    request: Option<u64>,
) -> Result<SemanticWireRecord, Box<dyn Error>> {
    let mut matches = records.iter().filter(|record| {
        record.kind == kind && request.is_none_or(|request| record.request_id == request)
    });
    let value = matches
        .next()
        .copied()
        .ok_or("required lifecycle boundary is absent")?;
    if matches.next().is_some() {
        return Err("required lifecycle boundary is duplicated".into());
    }
    Ok(value)
}

fn exactly_one_peer(
    records: &[SemanticWireRecord],
    kind: u8,
    peer: u64,
) -> Result<SemanticWireRecord, Box<dyn Error>> {
    let mut matches = records
        .iter()
        .filter(|record| record.kind == kind && record.sequence_id == peer);
    let value = matches
        .next()
        .copied()
        .ok_or("EngineCore admission alias is absent")?;
    if matches.next().is_some() {
        return Err("EngineCore admission alias is duplicated".into());
    }
    Ok(value)
}

fn selected(
    records: &[SemanticWireRecord],
    kind: u8,
    request: u64,
) -> Result<Vec<SemanticWireRecord>, Box<dyn Error>> {
    let count = records
        .iter()
        .filter(|record| record.kind == kind && record.request_id == request)
        .count();
    let mut output = Vec::new();
    output.try_reserve_exact(count)?;
    output.extend(
        records
            .iter()
            .filter(|record| record.kind == kind && record.request_id == request)
            .copied(),
    );
    Ok(output)
}

fn require_order(label: &str, earlier: u64, later: u64) -> Result<(), Box<dyn Error>> {
    if later < earlier {
        return Err(format!("negative interval {label}: {earlier} -> {later}").into());
    }
    Ok(())
}
