use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use gpu_observer_collector::join_b_trace::{load_semantic, OwnershipSlice, SemanticTrace};
use gpu_observer_collector::msgpack;
use serde::Serialize;
use serde_json::Value;

const QUERY_MAGIC: &[u8; 8] = b"GOQRY01\0";
const QUERY_VERSION: u32 = 1;
const QUERY_HEADER_BYTES: usize = 24;
const MAX_TEXT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_QUERY_FILE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_CUPTI_RECORDS: usize = 1_000_000;

#[derive(Clone, Copy)]
struct ClockPair {
    cupti_ns: u64,
    monotonic_ns: u64,
    uncertainty_ns: u64,
}

#[derive(Clone, Copy)]
struct ClockMap {
    start: ClockPair,
    end: ClockPair,
}

impl ClockMap {
    fn normalize(self, timestamp_ns: u64) -> Result<u64, Box<dyn Error>> {
        if self.start.cupti_ns >= self.end.cupti_ns
            || self.start.monotonic_ns >= self.end.monotonic_ns
            || timestamp_ns < self.start.cupti_ns
            || timestamp_ns > self.end.cupti_ns
        {
            return Err("CUPTI timestamp is outside a valid calibration interval".into());
        }
        let x = u128::from(timestamp_ns - self.start.cupti_ns);
        let x_span = u128::from(self.end.cupti_ns - self.start.cupti_ns);
        let y_span = u128::from(self.end.monotonic_ns - self.start.monotonic_ns);
        let delta = x
            .checked_mul(y_span)
            .ok_or("clock normalization overflow")?
            / x_span;
        u64::try_from(u128::from(self.start.monotonic_ns) + delta)
            .map_err(|_| "normalized timestamp overflow".into())
    }
}

#[derive(Clone)]
struct KernelActivity {
    start_ns: u64,
    end_ns: u64,
    stream: u32,
    correlation: u32,
    grid_id: u64,
    graph_node_id: u64,
    graph_id: u32,
    grid: [u32; 3],
    block: [u32; 3],
    name: String,
}

#[derive(Clone, Copy)]
struct RuntimeActivity {
    start_ns: u64,
    end_ns: u64,
    process: u32,
    thread: u32,
    correlation: u32,
    cbid: u32,
    return_value: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TraceBundle {
    schema: &'static str,
    generated_by: &'static str,
    run: Value,
    query: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_manifest: Option<Value>,
    focus: Focus,
    evidence: Evidence,
    metrics: Metrics,
    steps: Vec<StepView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deep_replay: Option<Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Focus {
    external_request_id: String,
    internal_request_id: String,
    request_hash: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Evidence {
    measured: Vec<&'static str>,
    reconstructed: Vec<&'static str>,
    matched_replay: bool,
    matched: Vec<&'static str>,
    unavailable: Vec<&'static str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Metrics {
    semantic_records: u64,
    semantic_loss_markers: u64,
    engine_steps: usize,
    packed_steps: u64,
    scheduler_order_mismatch_steps: u64,
    cupti_kernels: usize,
    clock_start_uncertainty_ns: u64,
    clock_end_uncertainty_ns: u64,
    clock_offset_drift_ns: i128,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StepView {
    id: u64,
    phase: &'static str,
    begin_ns: String,
    packed_ns: Option<String>,
    end_ns: String,
    wall_us: f64,
    scheduled_tokens: u32,
    prefill_tokens: u32,
    decode_tokens: u32,
    queue_depth: u32,
    active_requests: u32,
    kv_cache_usage_permyriad: u16,
    scheduler_order_mismatches: u32,
    scheduler_slices: Vec<SchedulerSliceView>,
    packed_slices: Vec<PackedSliceView>,
    focused_packed_tokens: Vec<PackedTokenView>,
    accepted_output_tokens: Vec<OutputTokenView>,
    kernels: Vec<KernelView>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SchedulerSliceView {
    request_id: String,
    phase: &'static str,
    scheduled_tokens: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PackedSliceView {
    request_id: String,
    packed_index: u32,
    row_begin: u32,
    row_end: u32,
    phase: &'static str,
    scheduled_tokens: u32,
    authoritative: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PackedTokenView {
    request_id: String,
    packing_generation: u64,
    packed_row: u32,
    sequence_position: u32,
    token_id: u32,
    phase: &'static str,
    timestamp_ns: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputTokenView {
    request_id: String,
    output_position: u64,
    token_id: u32,
    timestamp_ns: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct KernelView {
    name: String,
    correlation_id: u32,
    runtime_submit_ns: String,
    runtime_submit_from_step_us: f64,
    gpu_start_ns: String,
    gpu_end_ns: String,
    start_from_step_us: f64,
    end_from_step_us: f64,
    duration_us: f64,
    process: u32,
    thread: u32,
    runtime_cbid: u32,
    stream: u32,
    grid_id: u64,
    graph_node_id: u64,
    graph_id: u32,
    grid: [u32; 3],
    block: [u32; 3],
    request_block_ownership: Vec<BlockOwnershipView>,
    padding_blocks: u32,
    attribution: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BlockOwnershipView {
    request_id: String,
    block_begin: u32,
    block_end: u32,
    phase: &'static str,
    basis: &'static str,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (query, output_manifest) = load_query_frames(path_arg(&args, "--query")?)?;
    let run = load_json(path_arg(&args, "--run")?)?;
    let semantic = load_semantic(path_arg(&args, "--semantic")?)?;
    if semantic.loss_markers != 0 {
        return Err("refusing lossy semantic trace".into());
    }

    let internal_request_id = required_string(&query, "internal_request_id")?.to_owned();
    let external_request_id = required_string(&query, "external_request_id")?.to_owned();
    let request_hash = stable_request_id(&internal_request_id);
    if !semantic
        .slices
        .iter()
        .any(|record| record.request_id == request_hash)
    {
        return Err(format!(
            "focused request {internal_request_id} ({request_hash:#018x}) is absent from semantic membership"
        )
        .into());
    }

    validate_prompt_tokens(&query, &semantic, request_hash)?;
    if let Some(output) = &output_manifest {
        validate_output_tokens(output, &semantic, request_hash)?;
    }

    let kernels = load_kernels(path_arg(&args, "--activities")?)?;
    let runtimes = load_runtimes(path_arg(&args, "--runtime")?)?;
    let (clock, _quality) = load_clock(path_arg(&args, "--summary")?)?;
    let mut runtime_by_correlation = BTreeMap::new();
    for runtime in runtimes {
        if runtime.return_value != 0 {
            return Err(format!(
                "CUDA runtime activity {} returned {}",
                runtime.correlation, runtime.return_value
            )
            .into());
        }
        if runtime_by_correlation
            .insert(runtime.correlation, runtime)
            .is_some()
        {
            return Err(format!("duplicate CUPTI correlation {}", runtime.correlation).into());
        }
    }

    let assigned = assign_kernels(&semantic, kernels, &runtime_by_correlation, clock)?;
    let deep_replay = if let Some(path) = args.get("--deep") {
        let deep = load_json(Path::new(path))?;
        validate_replay(&run, &deep)?;
        Some(deep)
    } else {
        None
    };

    let mut steps = Vec::new();
    steps.try_reserve_exact(semantic.steps.len())?;
    let mut kernel_count = 0_usize;
    for (step_index, step) in semantic.steps.iter().enumerate() {
        let scheduler = &semantic.slices[step.slice_start..step.slice_start + step.slice_count];
        let packed = &semantic.packed_slices
            [step.packed_slice_start..step.packed_slice_start + step.packed_slice_count];
        let focused_tokens = &semantic.packed_tokens
            [step.packed_token_start..step.packed_token_start + step.packed_token_count];
        let output_tokens = &semantic.output_tokens
            [step.output_token_start..step.output_token_start + step.output_token_count];
        let owners = &semantic.owners[step.owner_start..step.owner_start + step.owner_count];
        let step_kernels = assigned.get(&step_index).map(Vec::as_slice).unwrap_or(&[]);
        kernel_count += step_kernels.len();

        steps.push(StepView {
            id: step.begin.step_id,
            phase: step_phase(step.begin.prefill_tokens, step.begin.decode_tokens),
            begin_ns: step.begin.timestamp_ns.to_string(),
            packed_ns: step
                .packed_begin
                .map(|record| record.timestamp_ns.to_string()),
            end_ns: step.end_timestamp_ns.to_string(),
            wall_us: delta_us(step.end_timestamp_ns, step.begin.timestamp_ns),
            scheduled_tokens: step.begin.scheduled_tokens,
            prefill_tokens: step.begin.prefill_tokens,
            decode_tokens: step.begin.decode_tokens,
            queue_depth: step.begin.queue_depth,
            active_requests: step.begin.active_requests,
            kv_cache_usage_permyriad: step.begin.kv_cache_usage_permyriad,
            scheduler_order_mismatches: step.scheduler_order_mismatches,
            scheduler_slices: scheduler
                .iter()
                .map(|record| SchedulerSliceView {
                    request_id: hex_id(record.request_id),
                    phase: phase(record.phase),
                    scheduled_tokens: record.scheduled_tokens,
                })
                .collect(),
            packed_slices: packed
                .iter()
                .map(|record| PackedSliceView {
                    request_id: hex_id(record.request_id),
                    packed_index: record.queue_depth,
                    row_begin: record.prefill_tokens,
                    row_end: record.decode_tokens,
                    phase: phase(record.phase),
                    scheduled_tokens: record.scheduled_tokens,
                    authoritative: true,
                })
                .collect(),
            focused_packed_tokens: focused_tokens
                .iter()
                .map(|record| PackedTokenView {
                    request_id: hex_id(record.request_id),
                    packing_generation: record.sequence_id,
                    packed_row: record.prefill_tokens,
                    sequence_position: record.decode_tokens,
                    token_id: record.scheduled_tokens,
                    phase: phase(record.phase),
                    timestamp_ns: record.timestamp_ns.to_string(),
                })
                .collect(),
            accepted_output_tokens: output_tokens
                .iter()
                .map(|record| OutputTokenView {
                    request_id: hex_id(record.request_id),
                    output_position: record.sequence_id,
                    token_id: record.scheduled_tokens,
                    timestamp_ns: record.timestamp_ns.to_string(),
                })
                .collect(),
            kernels: step_kernels
                .iter()
                .map(|(kernel, runtime, submit_ns, start_ns, end_ns)| {
                    kernel_view(
                        kernel,
                        runtime,
                        *submit_ns,
                        *start_ns,
                        *end_ns,
                        step.begin.timestamp_ns,
                        owners,
                    )
                })
                .collect(),
        });
    }

    let offset_start = i128::from(clock.start.monotonic_ns) - i128::from(clock.start.cupti_ns);
    let offset_end = i128::from(clock.end.monotonic_ns) - i128::from(clock.end.cupti_ns);
    let mut measured = vec![
        "OpenAI request, rendered prompt, tokenizer output, and prompt token IDs",
        "EngineCore scheduler membership and step boundaries",
        "GPUModelRunner authoritative packed rows and focused token positions",
        "Frontend prompt token IDs exactly match authoritative packed prefill tokens",
        "CUPTI runtime correlation IDs and actual GPU kernel intervals",
    ];
    let mut unavailable = vec![
        "PTX for shipped Qwen3-14B BF16 FlashAttention and vLLM operator binaries",
        "Exact per-request ownership for kernel families without a validated row-to-block rule",
    ];
    if output_manifest.is_some() {
        measured.push("Completed response text, output token IDs, and tokenizer strings");
        measured.push("Frontend output token IDs exactly match EngineCore accepted tokens");
    } else {
        unavailable.push("Frontend output-token strings; no output manifest was captured");
    }

    let bundle = TraceBundle {
        schema: "GPU_OBSERVER_TRACE_V2",
        generated_by: "gpu-observer export_trace_bundle",
        run,
        query,
        output_manifest,
        focus: Focus {
            external_request_id,
            internal_request_id,
            request_hash: hex_id(request_hash),
        },
        evidence: Evidence {
            measured,
            reconstructed: vec![
                "Focused request identity from the shared stable 64-bit hash",
                "Cache-kernel block ownership from authoritative half-open packed rows",
            ],
            matched_replay: deep_replay.is_some(),
            matched: if deep_replay.is_some() {
                vec![
                    "Compute Sanitizer device PC samples mapped to exact SASS offsets",
                    "Replay fingerprint and canonical semantic step signature match the timed run",
                ]
            } else {
                Vec::new()
            },
            unavailable,
        },
        metrics: Metrics {
            semantic_records: semantic.records,
            semantic_loss_markers: semantic.loss_markers,
            engine_steps: semantic.steps.len(),
            packed_steps: semantic.packed_steps,
            scheduler_order_mismatch_steps: semantic.scheduler_order_mismatch_steps,
            cupti_kernels: kernel_count,
            clock_start_uncertainty_ns: clock.start.uncertainty_ns,
            clock_end_uncertainty_ns: clock.end.uncertainty_ns,
            clock_offset_drift_ns: offset_end - offset_start,
        },
        steps,
        deep_replay,
    };

    let output_path = path_arg(&args, "--output")?;
    let output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output_path)?;
    let mut output = BufWriter::with_capacity(1024 * 1024, output);
    serde_json::to_writer_pretty(&mut output, &bundle)?;
    output.write_all(b"\n")?;
    output.flush()?;
    output.get_ref().sync_data()?;
    println!(
        "trace_bundle schema=GPU_OBSERVER_TRACE_V2 steps={} kernels={} focus={:#018x} output={}",
        semantic.steps.len(),
        kernel_count,
        request_hash,
        output_path.display()
    );
    Ok(())
}

type AssignedKernel = (KernelActivity, RuntimeActivity, u64, u64, u64);

fn assign_kernels(
    semantic: &SemanticTrace,
    kernels: Vec<KernelActivity>,
    runtimes: &BTreeMap<u32, RuntimeActivity>,
    clock: ClockMap,
) -> Result<BTreeMap<usize, Vec<AssignedKernel>>, Box<dyn Error>> {
    let mut timeline = Vec::new();
    for (index, step) in semantic.steps.iter().enumerate() {
        let packed = step
            .packed_begin
            .ok_or("every exported step must have authoritative packed rows")?;
        timeline.push((packed.timestamp_ns, index));
    }
    timeline.sort_unstable();
    if timeline.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err("packed-layout timestamps are not strictly increasing".into());
    }

    let mut assigned: BTreeMap<usize, Vec<AssignedKernel>> = BTreeMap::new();
    for kernel in kernels {
        let runtime = runtimes
            .get(&kernel.correlation)
            .copied()
            .ok_or_else(|| format!("missing runtime correlation {}", kernel.correlation))?;
        let submit_ns = clock.normalize(runtime.start_ns)?;
        let owner = timeline
            .iter()
            .rev()
            .find(|(packed_ns, index)| {
                *packed_ns <= submit_ns && submit_ns <= semantic.steps[*index].end_timestamp_ns
            })
            .map(|(_, index)| *index)
            .ok_or_else(|| {
                format!(
                    "kernel correlation {} cannot be bracketed by an engine step",
                    kernel.correlation
                )
            })?;
        let gpu_start_ns = clock.normalize(kernel.start_ns)?;
        let gpu_end_ns = clock.normalize(kernel.end_ns)?;
        assigned.entry(owner).or_default().push((
            kernel,
            runtime,
            submit_ns,
            gpu_start_ns,
            gpu_end_ns,
        ));
    }
    for values in assigned.values_mut() {
        values.sort_unstable_by_key(|(_, _, _, start, end)| (*start, *end));
    }
    Ok(assigned)
}

fn kernel_view(
    kernel: &KernelActivity,
    runtime: &RuntimeActivity,
    submit_ns: u64,
    gpu_start_ns: u64,
    gpu_end_ns: u64,
    step_begin_ns: u64,
    owners: &[OwnershipSlice],
) -> KernelView {
    let validated_cache_mapping = kernel.name.contains("reshape_and_cache_flash_kernel")
        && kernel.grid[1] == 1
        && kernel.grid[2] == 1;
    let ownership = if validated_cache_mapping {
        owners
            .iter()
            .filter_map(|owner| {
                let block_end = owner.row_end.min(kernel.grid[0]);
                (owner.row_begin < block_end).then(|| BlockOwnershipView {
                    request_id: hex_id(owner.request_id),
                    block_begin: owner.row_begin,
                    block_end,
                    phase: phase(owner.phase),
                    basis: if owner.authoritative {
                        "authoritative_packed_rows"
                    } else {
                        "scheduler_order_prediction"
                    },
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    KernelView {
        name: kernel.name.clone(),
        correlation_id: kernel.correlation,
        runtime_submit_ns: submit_ns.to_string(),
        runtime_submit_from_step_us: signed_delta_us(submit_ns, step_begin_ns),
        gpu_start_ns: gpu_start_ns.to_string(),
        gpu_end_ns: gpu_end_ns.to_string(),
        start_from_step_us: signed_delta_us(gpu_start_ns, step_begin_ns),
        end_from_step_us: signed_delta_us(gpu_end_ns, step_begin_ns),
        duration_us: delta_us(gpu_end_ns, gpu_start_ns),
        process: runtime.process,
        thread: runtime.thread,
        runtime_cbid: runtime.cbid,
        stream: kernel.stream,
        grid_id: kernel.grid_id,
        graph_node_id: kernel.graph_node_id,
        graph_id: kernel.graph_id,
        grid: kernel.grid,
        block: kernel.block,
        request_block_ownership: ownership,
        padding_blocks: kernel.grid[0]
            .saturating_sub(owners.iter().map(|owner| owner.row_end).max().unwrap_or(0)),
        attribution: if validated_cache_mapping {
            "validated_cache_row_to_block"
        } else {
            "step_many_to_many_only"
        },
    }
}

fn parse_args() -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut values = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(name) = args.next() {
        if !matches!(
            name.as_str(),
            "--query"
                | "--semantic"
                | "--activities"
                | "--runtime"
                | "--summary"
                | "--run"
                | "--output"
                | "--deep"
        ) {
            return Err(usage().into());
        }
        let value = args.next().ok_or_else(usage)?;
        if values.insert(name, value).is_some() {
            return Err("duplicate command-line option".into());
        }
    }
    for required in [
        "--query",
        "--semantic",
        "--activities",
        "--runtime",
        "--summary",
        "--run",
        "--output",
    ] {
        if !values.contains_key(required) {
            return Err(usage().into());
        }
    }
    Ok(values)
}

fn usage() -> String {
    "usage: export_trace_bundle --query QUERY.frames --semantic SEMANTIC.bin \
--activities CUPTI.activities.tsv --runtime CUPTI.runtime.tsv \
--summary CUPTI.summary.tsv --run RUN.json --output TRACE.json [--deep DEEP.json]"
        .to_owned()
}

fn path_arg<'a>(
    args: &'a BTreeMap<String, String>,
    name: &str,
) -> Result<&'a Path, Box<dyn Error>> {
    args.get(name).map(Path::new).ok_or_else(|| usage().into())
}

fn load_query_frames(path: &Path) -> Result<(Value, Option<Value>), Box<dyn Error>> {
    let mut file = bounded_file(path, MAX_QUERY_FILE_BYTES)?;
    let mut header = [0_u8; QUERY_HEADER_BYTES];
    file.read_exact(&mut header)?;
    if &header[0..8] != QUERY_MAGIC
        || u32::from_le_bytes(header[8..12].try_into()?) != QUERY_VERSION
        || header[16..24] != [0; 8]
    {
        return Err("invalid query frame header".into());
    }
    let frame_bound = u32::from_le_bytes(header[12..16].try_into()?) as usize;
    if frame_bound == 0 || frame_bound > msgpack::MAX_MESSAGEPACK_BYTES {
        return Err("invalid query frame bound".into());
    }

    let mut query = None;
    let mut output = None;
    let mut frame_count = 0_u8;
    loop {
        let mut raw_length = [0_u8; 4];
        let read = file.read(&mut raw_length)?;
        if read == 0 {
            break;
        }
        if read != raw_length.len() {
            return Err("truncated query frame length".into());
        }
        frame_count = frame_count
            .checked_add(1)
            .ok_or("query frame count overflow")?;
        if frame_count > 8 {
            return Err("query capture exceeds eight bounded frames".into());
        }
        let length = u32::from_le_bytes(raw_length) as usize;
        if length == 0 || length > frame_bound {
            return Err("invalid query manifest length".into());
        }
        let mut payload = Vec::new();
        payload.try_reserve_exact(length)?;
        payload.resize(length, 0);
        file.read_exact(&mut payload)?;
        let value = msgpack::decode(&payload)?;
        if required_string(&value, "schema")? != "GPU_OBSERVER_QUERY_01" {
            return Err("unsupported query manifest schema".into());
        }
        match value.get("kind").and_then(Value::as_str).unwrap_or("query") {
            "query" if query.is_none() => query = Some(value),
            "output" if output.is_none() => output = Some(value),
            "query" | "output" => return Err("duplicate focused query frame kind".into()),
            _ => return Err("unknown focused query frame kind".into()),
        }
    }
    let query = query.ok_or("query capture has no query manifest")?;
    if let Some(output_value) = &output {
        if required_string(output_value, "internal_request_id")?
            != required_string(&query, "internal_request_id")?
            || required_string(output_value, "external_request_id")?
                != required_string(&query, "external_request_id")?
        {
            return Err("output manifest belongs to a different request".into());
        }
    }
    Ok((query, output))
}

fn manifest_token_ids(
    value: &Value,
    ids_key: &str,
    tokens_key: &str,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let ids = value
        .get(ids_key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("manifest field {ids_key} is missing or not an array"))?;
    let tokens = value
        .get(tokens_key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("manifest field {tokens_key} is missing or not an array"))?;
    if ids.len() != tokens.len() {
        return Err(format!("manifest {ids_key} and {tokens_key} lengths differ").into());
    }

    let mut output = Vec::new();
    output.try_reserve_exact(ids.len())?;
    for (index, (id, token)) in ids.iter().zip(tokens).enumerate() {
        let id = id
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("manifest {ids_key}[{index}] is not a u32"))?;
        let token_id = token
            .get("id")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("manifest {tokens_key}[{index}].id is not a u32"))?;
        let position = token
            .get("position")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("manifest {tokens_key}[{index}].position is invalid"))?;
        if position != index as u64 || token_id != id {
            return Err(format!("manifest token metadata disagrees at position {index}").into());
        }
        if !token.get("raw").is_some_and(Value::is_string)
            || token
                .get("display")
                .is_none_or(|display| !display.is_null() && !display.is_string())
        {
            return Err(format!("manifest token strings are invalid at position {index}").into());
        }
        output.push(id);
    }
    Ok(output)
}

fn validate_exact_sequence(
    label: &str,
    expected: &[u32],
    mut observed: Vec<(u64, u32)>,
) -> Result<(), Box<dyn Error>> {
    observed.sort_unstable();
    if observed.len() != expected.len() {
        return Err(format!(
            "{label} token count differs: manifest={} observed={}",
            expected.len(),
            observed.len()
        )
        .into());
    }
    for (index, (position, token_id)) in observed.into_iter().enumerate() {
        if position != index as u64 || token_id != expected[index] {
            return Err(format!("{label} token mismatch at position {index}").into());
        }
    }
    Ok(())
}

fn validate_prompt_tokens(
    query: &Value,
    semantic: &SemanticTrace,
    request_hash: u64,
) -> Result<(), Box<dyn Error>> {
    let expected = manifest_token_ids(query, "prompt_token_ids", "tokens")?;
    let observed = semantic
        .packed_tokens
        .iter()
        .filter(|record| record.request_id == request_hash && record.phase == 1)
        .map(|record| (u64::from(record.decode_tokens), record.scheduled_tokens))
        .collect();
    validate_exact_sequence("prompt-to-packed", &expected, observed)
}

fn validate_output_tokens(
    output: &Value,
    semantic: &SemanticTrace,
    request_hash: u64,
) -> Result<(), Box<dyn Error>> {
    let choices = output
        .get("choices")
        .and_then(Value::as_array)
        .ok_or("output manifest choices are missing")?;
    let [choice] = choices.as_slice() else {
        return Err("focused output manifest must contain exactly one choice".into());
    };
    let expected = manifest_token_ids(choice, "token_ids", "tokens")?;
    let observed = semantic
        .output_tokens
        .iter()
        .filter(|record| record.request_id == request_hash)
        .map(|record| (record.sequence_id, record.scheduled_tokens))
        .collect();
    validate_exact_sequence("accepted-to-output", &expected, observed)
}

fn load_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    let file = bounded_file(path, MAX_TEXT_BYTES)?;
    let value: Value = serde_json::from_reader(BufReader::new(file))?;
    if !value.is_object() {
        return Err(format!("{} must contain a JSON object", path.display()).into());
    }
    Ok(value)
}

fn validate_replay(run: &Value, deep: &Value) -> Result<(), Box<dyn Error>> {
    for field in ["replayFingerprint", "semanticSignature"] {
        if required_string(run, field)? != required_string(deep, field)? {
            return Err(format!("deep replay {field} does not match the timed run").into());
        }
    }
    Ok(())
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Box<dyn Error>> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("manifest field {key} is missing or not a string").into())
}

fn load_kernels(path: &Path) -> Result<Vec<KernelActivity>, Box<dyn Error>> {
    let file = bounded_file(path, MAX_TEXT_BYTES)?;
    let mut format = false;
    let mut header = false;
    let mut output = Vec::new();
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
        if output.len() == MAX_CUPTI_RECORDS {
            return Err("CUPTI kernel trace exceeds one million records".into());
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 18 {
            return Err(format!("malformed kernel row {}", line_number + 1).into());
        }
        let row = KernelActivity {
            start_ns: fields[0].parse()?,
            end_ns: fields[1].parse()?,
            stream: fields[4].parse()?,
            correlation: fields[5].parse()?,
            grid_id: fields[6].parse()?,
            graph_node_id: fields[7].parse()?,
            graph_id: fields[8].parse()?,
            grid: [fields[9].parse()?, fields[10].parse()?, fields[11].parse()?],
            block: [
                fields[12].parse()?,
                fields[13].parse()?,
                fields[14].parse()?,
            ],
            name: fields[17].to_owned(),
        };
        if row.start_ns == 0
            || row.end_ns <= row.start_ns
            || row.correlation == 0
            || row.grid.contains(&0)
            || row.block.contains(&0)
            || row.name.is_empty()
        {
            return Err(format!("invalid kernel row {}", line_number + 1).into());
        }
        output.try_reserve(1)?;
        output.push(row);
    }
    if !format || !header || output.is_empty() {
        return Err("CUPTI kernel trace lacks its header or records".into());
    }
    Ok(output)
}

fn load_runtimes(path: &Path) -> Result<Vec<RuntimeActivity>, Box<dyn Error>> {
    let file = bounded_file(path, MAX_TEXT_BYTES)?;
    let mut format = false;
    let mut header = false;
    let mut output = Vec::new();
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
        if output.len() == MAX_CUPTI_RECORDS {
            return Err("CUPTI runtime trace exceeds one million records".into());
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 7 {
            return Err(format!("malformed runtime row {}", line_number + 1).into());
        }
        let row = RuntimeActivity {
            start_ns: fields[0].parse()?,
            end_ns: fields[1].parse()?,
            process: fields[2].parse()?,
            thread: fields[3].parse()?,
            correlation: fields[4].parse()?,
            cbid: fields[5].parse()?,
            return_value: fields[6].parse()?,
        };
        if row.start_ns == 0 || row.end_ns < row.start_ns || row.correlation == 0 {
            return Err(format!("invalid runtime row {}", line_number + 1).into());
        }
        output.try_reserve(1)?;
        output.push(row);
    }
    if !format || !header || output.is_empty() {
        return Err("CUPTI runtime trace lacks its header or records".into());
    }
    Ok(output)
}

fn load_clock(path: &Path) -> Result<(ClockMap, Value), Box<dyn Error>> {
    let file = bounded_file(path, 1024 * 1024)?;
    let lines: Vec<String> = BufReader::new(file).lines().collect::<Result<_, _>>()?;
    if lines
        .first()
        .is_none_or(|line| !line.starts_with("# format=GOCUPTI01 "))
    {
        return Err("CUPTI summary lacks its versioned header".into());
    }
    let pair = |name: &str| -> Result<ClockPair, Box<dyn Error>> {
        let line = lines
            .iter()
            .find(|line| line.starts_with(name) && line.as_bytes().get(name.len()) == Some(&b'\t'))
            .ok_or_else(|| format!("CUPTI summary lacks {name}"))?;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 5 {
            return Err(format!("malformed CUPTI clock row {name}").into());
        }
        Ok(ClockPair {
            cupti_ns: fields[1].parse()?,
            monotonic_ns: fields[2].parse()?,
            uncertainty_ns: fields[3].parse()?,
        })
    };
    let clock = ClockMap {
        start: pair("CLOCK_MONOTONIC_START")?,
        end: pair("CLOCK_MONOTONIC_END")?,
    };
    if clock.start.cupti_ns == 0
        || clock.end.cupti_ns <= clock.start.cupti_ns
        || clock.end.monotonic_ns <= clock.start.monotonic_ns
    {
        return Err("invalid CUPTI clock calibration".into());
    }

    let status_header = lines
        .get(1)
        .ok_or("CUPTI summary status header is missing")?
        .split('\t');
    let status_values = lines
        .get(2)
        .ok_or("CUPTI summary status row is missing")?
        .split('\t');
    let quality = Value::Object(
        status_header
            .zip(status_values)
            .map(|(key, value)| (key.to_owned(), Value::String(value.to_owned())))
            .collect(),
    );
    Ok((clock, quality))
}

fn bounded_file(path: &Path, max_bytes: u64) -> Result<File, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > max_bytes {
        return Err(format!(
            "{} has invalid bounded size {} (max {})",
            path.display(),
            bytes,
            max_bytes
        )
        .into());
    }
    Ok(file)
}

fn stable_request_id(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn phase(value: u8) -> &'static str {
    match value {
        1 => "prefill",
        2 => "decode",
        _ => "unknown",
    }
}

fn step_phase(prefill: u32, decode: u32) -> &'static str {
    match (prefill != 0, decode != 0) {
        (true, false) => "prefill",
        (false, true) => "decode",
        (true, true) => "mixed",
        (false, false) => "empty",
    }
}

fn hex_id(value: u64) -> String {
    format!("0x{value:016x}")
}

fn delta_us(end: u64, start: u64) -> f64 {
    end.saturating_sub(start) as f64 / 1_000.0
}

fn signed_delta_us(value: u64, origin: u64) -> f64 {
    (i128::from(value) - i128::from(origin)) as f64 / 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_request_hash_matches_python_fnv1a() {
        assert_eq!(
            stable_request_id("chatcmpl-semantic-smoke"),
            0xef3d06865cf2f3c6
        );
    }

    #[test]
    fn affine_clock_map_handles_offset_drift() {
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
    fn exact_token_sequence_rejects_reordering_and_missing_tokens() {
        assert!(validate_exact_sequence("test", &[10, 20], vec![(1, 20), (0, 10)]).is_ok());
        assert!(validate_exact_sequence("test", &[10, 20], vec![(0, 20), (1, 10)]).is_err());
        assert!(validate_exact_sequence("test", &[10, 20], vec![(0, 10)]).is_err());
    }
}
