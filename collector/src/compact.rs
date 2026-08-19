//! Cold translation between debuggable JSON events and compact core records.

use std::collections::{BTreeMap, BTreeSet};

use gpu_observer_core as hot;

use crate::error::{ObserverError, Result};
use crate::event::{Event, EventKind, RequestPhase};
use crate::report::{
    CorrelatedStep, CorrelationReport, Diagnostic, DiagnosticSeverity, KernelExecution,
    RequestExecution, UnassignedEvent,
};

struct NameTable<'a> {
    names: Vec<&'a str>,
}

impl<'a> NameTable<'a> {
    fn from_values(mut values: Vec<&'a str>, max_names: usize) -> Result<Self> {
        values.sort_unstable();
        values.dedup();
        if values.len() > max_names {
            return Err(ObserverError::InvalidEvent {
                event_id: "collector".to_owned(),
                message: format!("identifier table exceeds {max_names} entries"),
            });
        }
        Ok(Self { names: values })
    }

    fn id(&self, name: &str) -> usize {
        self.names.binary_search(&name).unwrap() + 1
    }

    fn name(&self, id: usize) -> Option<&str> {
        id.checked_sub(1)
            .and_then(|index| self.names.get(index).copied())
    }
}

struct CompactInput<'a> {
    records: Vec<hot::EventRecord>,
    sequence_event_ids: Vec<&'a str>,
    request_names: NameTable<'a>,
    service_classes: NameTable<'a>,
    kernel_symbols: NameTable<'a>,
}

pub fn correlate(events: &[Event]) -> CorrelationReport {
    match correlate_checked(events) {
        Ok(report) => report,
        Err(error) => {
            let mut report = CorrelationReport::empty(events.len());
            report.diagnostics.push(Diagnostic {
                severity: DiagnosticSeverity::Error,
                code: "compact_conversion_failed".to_owned(),
                message: error.to_string(),
                event_ids: Vec::new(),
            });
            report
        }
    }
}

pub fn correlate_checked(events: &[Event]) -> Result<CorrelationReport> {
    let mut compact = compact_events(events)?;
    let trace =
        hot::correlate(&mut compact.records).map_err(|error| ObserverError::InvalidEvent {
            event_id: "collector".to_owned(),
            message: format!("core correlation failed: {error:?}"),
        })?;

    let mut report = CorrelationReport::empty(events.len());
    report.core_index_bytes = trace.allocated_bytes_lower_bound();
    report.steps = build_steps(&trace, &compact);
    report.requests = build_requests(events, &report.steps);
    append_diagnostics(&trace, &compact, &mut report);
    Ok(report)
}

fn compact_events<'a>(events: &'a [Event]) -> Result<CompactInput<'a>> {
    let expanded_count = events.iter().fold(events.len(), |total, event| {
        total
            + match &event.kind {
                EventKind::EngineStepBegin { request_slices, .. } => request_slices.len(),
                _ => 0,
            }
    });
    let (request_names, service_classes, kernel_symbols) =
        build_name_tables(events, expanded_count)?;

    let mut records = Vec::new();
    records
        .try_reserve_exact(expanded_count)
        .map_err(|_| allocation_error())?;
    let mut sequence_event_ids = Vec::new();
    sequence_event_ids
        .try_reserve_exact(expanded_count)
        .map_err(|_| allocation_error())?;
    let mut next_sequence = 1_u64;

    for event in events {
        let source = source_kind(event)?;
        match &event.kind {
            EventKind::RequestReceived {
                request_id,
                service_class,
                input_tokens,
                requested_output_tokens,
            } => {
                let data = hot::RequestReceivedData {
                    request_id: request_names.id(request_id) as u64,
                    input_tokens: input_tokens.unwrap_or(0),
                    requested_output_tokens: requested_output_tokens.unwrap_or(0),
                    service_class_id: service_class
                        .as_deref()
                        .map(|name| service_classes.id(name) as u16)
                        .unwrap_or(0),
                    reserved: 0,
                };
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    0,
                    |context| hot::EventRecord::request_received(context, data),
                );
            }
            EventKind::RequestQueued {
                request_id,
                queue_depth,
            } => {
                let data = hot::RequestQueuedData {
                    request_id: request_names.id(request_id) as u64,
                    queue_depth: *queue_depth,
                    reserved: 0,
                };
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    0,
                    |context| hot::EventRecord::request_queued(context, data),
                );
            }
            EventKind::TokenEmitted {
                request_id,
                token_index,
            } => {
                let data = hot::TokenEmittedData {
                    request_id: request_names.id(request_id) as u64,
                    token_index: *token_index,
                    reserved: 0,
                };
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    0,
                    |context| hot::EventRecord::token_emitted(context, data),
                );
            }
            EventKind::RequestCompleted {
                request_id,
                input_tokens,
                output_tokens,
                ttft_ns,
                e2e_latency_ns,
            } => {
                let data = hot::RequestCompletedData {
                    request_id: request_names.id(request_id) as u64,
                    input_tokens: *input_tokens,
                    output_tokens: *output_tokens,
                    ttft_ns: ttft_ns.unwrap_or(0),
                    e2e_latency_ns: *e2e_latency_ns,
                };
                let flags = if ttft_ns.is_some() {
                    hot::EventFlags::HAS_TTFT
                } else {
                    0
                };
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    flags,
                    |context| hot::EventRecord::request_completed(context, data),
                );
            }
            EventKind::EngineStepBegin {
                step_id,
                request_slices,
                prefill_tokens,
                decode_tokens,
                scheduled_tokens,
                queue_depth,
                active_requests,
                kv_cache_usage,
            } => {
                let data = hot::EngineStepBeginData {
                    step_id: *step_id,
                    scheduled_tokens: *scheduled_tokens,
                    prefill_tokens: *prefill_tokens,
                    decode_tokens: *decode_tokens,
                    queue_depth: *queue_depth,
                    active_requests: active_requests.unwrap_or(0),
                    expected_slices: u32::try_from(request_slices.len()).map_err(|_| {
                        invalid(
                            event,
                            "engine step contains more than u32::MAX request slices",
                        )
                    })?,
                    kv_cache_usage_permyriad: kv_cache_usage.map(kv_usage_permyriad).unwrap_or(0),
                    reserved: 0,
                };
                let flags = if kv_cache_usage.is_some() {
                    hot::EventFlags::HAS_KV_CACHE_USAGE
                } else {
                    0
                };
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    flags,
                    |context| hot::EventRecord::engine_step_begin(context, data),
                );
                for slice in request_slices {
                    let data = hot::StepRequestSliceData {
                        step_id: *step_id,
                        request_id: request_names.id(&slice.request_id) as u64,
                        sequence_id: slice.sequence_id.unwrap_or(0),
                        scheduled_tokens: slice.scheduled_tokens,
                        service_class_id: slice
                            .service_class
                            .as_deref()
                            .map(|name| service_classes.id(name) as u16)
                            .unwrap_or(0),
                        phase: match slice.phase {
                            RequestPhase::Prefill => hot::RequestPhase::Prefill,
                            RequestPhase::Decode => hot::RequestPhase::Decode,
                        },
                        reserved: 0,
                    };
                    append_record(
                        &mut records,
                        &mut sequence_event_ids,
                        &mut next_sequence,
                        event,
                        source,
                        0,
                        |context| hot::EventRecord::step_request_slice(context, data),
                    );
                }
            }
            EventKind::EngineStepEnd { step_id, status } => {
                let data = hot::EngineStepEndData {
                    step_id: *step_id,
                    status: u8::from(status.as_deref().is_some_and(|value| value != "ok")),
                    reserved: [0; 7],
                };
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    0,
                    |context| hot::EventRecord::engine_step_end(context, data),
                );
            }
            EventKind::CudaKernelLaunch {
                correlation_id,
                stream,
                kernel_function,
                kernel_symbol,
                grid,
                block,
                shared_memory_bytes,
                graph_id,
            } => {
                let data = hot::CudaKernelLaunchData {
                    correlation_id: correlation_id.unwrap_or(0),
                    stream: *stream,
                    kernel_function: parse_pointer(event, kernel_function)?,
                    graph_id: graph_id.unwrap_or(0),
                    grid: hot::Dim3 {
                        x: grid.x,
                        y: grid.y,
                        z: grid.z,
                    },
                    block: hot::Dim3 {
                        x: block.x,
                        y: block.y,
                        z: block.z,
                    },
                    shared_memory_bytes: *shared_memory_bytes,
                    kernel_symbol_id: kernel_symbol
                        .as_deref()
                        .map(|name| kernel_symbols.id(name) as u32)
                        .unwrap_or(0),
                };
                let mut flags = 0;
                if correlation_id.is_some() {
                    flags |= hot::EventFlags::HAS_CORRELATION_ID;
                }
                if graph_id.is_some() {
                    flags |= hot::EventFlags::HAS_GRAPH_ID;
                }
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    flags,
                    |context| hot::EventRecord::cuda_kernel_launch(context, data),
                );
            }
            EventKind::CudaMemcpy {
                correlation_id,
                stream,
                bytes,
                kind,
                asynchronous,
            } => {
                let data = hot::CudaMemcpyData {
                    correlation_id: correlation_id.unwrap_or(0),
                    stream: stream.unwrap_or(0),
                    bytes: *bytes,
                    kind: memcpy_kind(kind),
                    asynchronous: u8::from(*asynchronous),
                    reserved: [0; 6],
                };
                let flags = if correlation_id.is_some() {
                    hot::EventFlags::HAS_CORRELATION_ID
                } else {
                    0
                };
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    flags,
                    |context| hot::EventRecord::cuda_memcpy(context, data),
                );
            }
            EventKind::CudaGraphLaunch {
                correlation_id,
                stream,
                graph_id,
            } => {
                let data = hot::CudaGraphLaunchData {
                    correlation_id: correlation_id.unwrap_or(0),
                    stream: *stream,
                    graph_id: *graph_id,
                };
                let mut flags = hot::EventFlags::HAS_GRAPH_ID;
                if correlation_id.is_some() {
                    flags |= hot::EventFlags::HAS_CORRELATION_ID;
                }
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    flags,
                    |context| hot::EventRecord::cuda_graph_launch(context, data),
                );
            }
            EventKind::CuptiKernelActivity {
                correlation_id,
                stream,
                kernel_name,
                duration_ns,
                device_id,
                context_id,
                graph_id,
            } => {
                let data = hot::CuptiKernelActivityData {
                    correlation_id: *correlation_id,
                    stream: *stream,
                    duration_ns: *duration_ns,
                    graph_id: graph_id.unwrap_or(0),
                    kernel_symbol_id: kernel_symbols.id(kernel_name) as u32,
                    device_id: device_id.unwrap_or(0),
                    context_id: context_id.unwrap_or(0),
                    reserved: 0,
                };
                let flags = if graph_id.is_some() {
                    hot::EventFlags::HAS_GRAPH_ID
                } else {
                    0
                };
                append_record(
                    &mut records,
                    &mut sequence_event_ids,
                    &mut next_sequence,
                    event,
                    source,
                    flags,
                    |context| hot::EventRecord::cupti_kernel_activity(context, data),
                );
            }
            EventKind::CudaApiCall { .. }
            | EventKind::MetricSample { .. }
            | EventKind::ControlDecision { .. } => {}
        }
    }

    Ok(CompactInput {
        records,
        sequence_event_ids,
        request_names,
        service_classes,
        kernel_symbols,
    })
}

fn append_record<'a>(
    records: &mut Vec<hot::EventRecord>,
    event_ids: &mut Vec<&'a str>,
    sequence: &mut u64,
    event: &'a Event,
    source: hot::SourceKind,
    extra_flags: u16,
    build: impl FnOnce(hot::EventContext) -> hot::EventRecord,
) {
    let normalized = event.is_clock_normalized();
    let context = hot::EventContext {
        source_timestamp_ns: event.timestamp_ns,
        timestamp_ns: event.correlation_timestamp_ns(),
        sequence: *sequence,
        pid: event.pid,
        tid: event.tid.unwrap_or(0),
        clock_id: clock_id(&event.clock_domain),
        flags: extra_flags
            | if normalized {
                hot::EventFlags::TIMESTAMP_NORMALIZED
            } else {
                0
            },
        source,
    };
    records.push(build(context));
    event_ids.push(&event.event_id);
    *sequence = sequence.wrapping_add(1);
}

fn build_steps(trace: &hot::TraceIndex, input: &CompactInput<'_>) -> Vec<CorrelatedStep> {
    trace
        .steps()
        .iter()
        .map(|step| {
            let request_slices = trace
                .memberships_for_step(step)
                .iter()
                .map(|membership| {
                    let slice = membership.slice;
                    crate::event::RequestSlice {
                        request_id: input
                            .request_names
                            .name(slice.request_id as usize)
                            .unwrap_or("<unknown-request>")
                            .to_owned(),
                        sequence_id: (slice.sequence_id != 0).then_some(slice.sequence_id),
                        phase: match slice.phase {
                            hot::RequestPhase::Prefill => RequestPhase::Prefill,
                            hot::RequestPhase::Decode => RequestPhase::Decode,
                        },
                        scheduled_tokens: slice.scheduled_tokens,
                        service_class: input
                            .service_classes
                            .name(slice.service_class_id as usize)
                            .map(str::to_owned),
                    }
                })
                .collect();
            let kernels = trace
                .kernels_for_step(step)
                .iter()
                .map(|kernel| KernelExecution {
                    launch_event_id: event_id(input, kernel.launch_sequence).to_owned(),
                    activity_event_id: (kernel.flags & hot::correlation::KernelFlags::HAS_ACTIVITY
                        != 0)
                        .then(|| event_id(input, kernel.activity_sequence).to_owned()),
                    correlation_id: (kernel.flags
                        & hot::correlation::KernelFlags::HAS_CORRELATION_ID
                        != 0)
                        .then_some(kernel.launch.correlation_id),
                    stream: kernel.launch.stream,
                    kernel_function: format!("0x{:x}", kernel.launch.kernel_function),
                    kernel_name: input
                        .kernel_symbols
                        .name(kernel.launch.kernel_symbol_id as usize)
                        .map(str::to_owned),
                    grid: crate::event::Dim3 {
                        x: kernel.launch.grid.x,
                        y: kernel.launch.grid.y,
                        z: kernel.launch.grid.z,
                    },
                    block: crate::event::Dim3 {
                        x: kernel.launch.block.x,
                        y: kernel.launch.block.y,
                        z: kernel.launch.block.z,
                    },
                    shared_memory_bytes: kernel.launch.shared_memory_bytes,
                    graph_id: (kernel.flags & hot::correlation::KernelFlags::HAS_GRAPH_ID != 0)
                        .then_some(kernel.launch.graph_id),
                    host_launch_timestamp_ns: kernel.host_launch_timestamp_ns,
                    gpu_start_timestamp_ns: (kernel.flags
                        & hot::correlation::KernelFlags::HAS_ACTIVITY
                        != 0)
                        .then_some(kernel.gpu_start_timestamp_ns),
                    gpu_duration_ns: (kernel.flags & hot::correlation::KernelFlags::HAS_ACTIVITY
                        != 0)
                        .then_some(kernel.gpu_duration_ns),
                })
                .collect();
            let open = step.state & hot::correlation::StepState::OPEN != 0;
            CorrelatedStep {
                step_id: step.step_id,
                pid: step.pid,
                begin_timestamp_ns: step.begin_timestamp_ns,
                end_timestamp_ns: (!open).then_some(step.end_timestamp_ns),
                wall_time_ns: (!open)
                    .then(|| step.end_timestamp_ns.checked_sub(step.begin_timestamp_ns))
                    .flatten(),
                request_slices,
                prefill_tokens: step.prefill_tokens,
                decode_tokens: step.decode_tokens,
                scheduled_tokens: step.scheduled_tokens,
                queue_depth: step.queue_depth,
                active_requests: Some(step.active_requests),
                kv_cache_usage: Some(step.kv_cache_usage_permyriad as f64 / 10_000.0),
                cuda_launch_count: step.kernel_count as usize,
                graph_launch_count: step.graph_launch_count as usize,
                kernels,
                unmatched_cuda_launches: step.unmatched_launch_count as usize,
                kernel_time_sum_ns: step.kernel_time_sum_ns,
                gpu_busy_time_ns: step.gpu_busy_time_ns,
                gpu_span_ns: step.gpu_span_ns,
            }
        })
        .collect()
}

#[derive(Default)]
struct RequestFacts {
    service_class: Option<String>,
    queued_at_ns: Option<u64>,
    ttft_ns: Option<u64>,
    e2e_latency_ns: Option<u64>,
}

fn build_requests(events: &[Event], steps: &[CorrelatedStep]) -> Vec<RequestExecution> {
    let mut facts: BTreeMap<String, RequestFacts> = BTreeMap::new();
    for event in events {
        match &event.kind {
            EventKind::RequestReceived {
                request_id,
                service_class,
                ..
            } => {
                facts.entry(request_id.clone()).or_default().service_class = service_class.clone();
            }
            EventKind::RequestQueued { request_id, .. } => {
                facts.entry(request_id.clone()).or_default().queued_at_ns =
                    Some(event.correlation_timestamp_ns());
            }
            EventKind::RequestCompleted {
                request_id,
                ttft_ns,
                e2e_latency_ns,
                ..
            } => {
                let request = facts.entry(request_id.clone()).or_default();
                request.ttft_ns = *ttft_ns;
                request.e2e_latency_ns = Some(*e2e_latency_ns);
            }
            _ => {}
        }
    }
    for step in steps {
        for slice in &step.request_slices {
            let request = facts.entry(slice.request_id.clone()).or_default();
            if request.service_class.is_none() {
                request.service_class = slice.service_class.clone();
            }
        }
    }

    facts
        .into_iter()
        .map(|(request_id, facts)| {
            let member_steps: Vec<&CorrelatedStep> = steps
                .iter()
                .filter(|step| {
                    step.request_slices
                        .iter()
                        .any(|slice| slice.request_id == request_id)
                })
                .collect();
            let shared_with_request_ids = member_steps
                .iter()
                .flat_map(|step| step.request_slices.iter())
                .filter(|slice| slice.request_id != request_id)
                .map(|slice| slice.request_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let queue_delay_ns = facts.queued_at_ns.and_then(|queued| {
                member_steps
                    .iter()
                    .map(|step| step.begin_timestamp_ns)
                    .min()?
                    .checked_sub(queued)
            });
            RequestExecution {
                request_id,
                service_class: facts.service_class,
                step_ids: member_steps.iter().map(|step| step.step_id).collect(),
                shared_with_request_ids,
                queue_delay_ns,
                ttft_ns: facts.ttft_ns,
                e2e_latency_ns: facts.e2e_latency_ns,
            }
        })
        .collect()
}

fn append_diagnostics(
    trace: &hot::TraceIndex,
    input: &CompactInput<'_>,
    report: &mut CorrelationReport,
) {
    for diagnostic in trace.diagnostics() {
        let code = format!("{:?}", diagnostic.code).to_ascii_lowercase();
        let event_ids = (diagnostic.sequence != 0)
            .then(|| event_id(input, diagnostic.sequence).to_owned())
            .into_iter()
            .collect();
        report.diagnostics.push(Diagnostic {
            severity: diagnostic_severity(diagnostic.code),
            code: code.clone(),
            message: format!(
                "{:?}: pid={}, step_id={}, sequence={}",
                diagnostic.code, diagnostic.pid, diagnostic.step_id, diagnostic.sequence
            ),
            event_ids,
        });
        match diagnostic.code {
            hot::correlation::DiagnosticCode::LaunchOutsideStep => {
                report.unassigned_events.push(UnassignedEvent {
                    event_id: event_id(input, diagnostic.sequence).to_owned(),
                    event: "cuda_kernel_or_graph_launch".to_owned(),
                    reason: "no open engine step existed for the same PID".to_owned(),
                });
            }
            hot::correlation::DiagnosticCode::ActivityWithoutLaunch => {
                report.unassigned_events.push(UnassignedEvent {
                    event_id: event_id(input, diagnostic.sequence).to_owned(),
                    event: "cupti_kernel_activity".to_owned(),
                    reason: "no launch matched (PID, stream, correlation_id)".to_owned(),
                });
            }
            _ => {}
        }
    }
}

fn diagnostic_severity(code: hot::correlation::DiagnosticCode) -> DiagnosticSeverity {
    match code {
        hot::correlation::DiagnosticCode::UnnormalizedClock
        | hot::correlation::DiagnosticCode::OpenStep
        | hot::correlation::DiagnosticCode::SliceCountMismatch
        | hot::correlation::DiagnosticCode::TelemetryLoss
        | hot::correlation::DiagnosticCode::ActivityWithoutLaunch => DiagnosticSeverity::Warning,
        _ => DiagnosticSeverity::Error,
    }
}

fn build_name_tables<'a>(
    events: &'a [Event],
    expanded_count: usize,
) -> Result<(NameTable<'a>, NameTable<'a>, NameTable<'a>)> {
    let mut requests = reserved_references(expanded_count)?;
    let mut service_classes = reserved_references(expanded_count)?;
    let mut kernel_symbols = reserved_references(events.len())?;

    for event in events {
        match &event.kind {
            EventKind::RequestReceived { request_id, .. }
            | EventKind::RequestQueued { request_id, .. }
            | EventKind::TokenEmitted { request_id, .. }
            | EventKind::RequestCompleted { request_id, .. } => {
                requests.push(request_id.as_str());
            }
            EventKind::EngineStepBegin { request_slices, .. } => {
                requests.extend(request_slices.iter().map(|slice| slice.request_id.as_str()));
            }
            _ => {}
        }

        match &event.kind {
            EventKind::RequestReceived {
                service_class: Some(service_class),
                ..
            } => service_classes.push(service_class.as_str()),
            EventKind::EngineStepBegin { request_slices, .. } => {
                service_classes.extend(
                    request_slices
                        .iter()
                        .filter_map(|slice| slice.service_class.as_deref()),
                );
            }
            _ => {}
        }

        match &event.kind {
            EventKind::CudaKernelLaunch {
                kernel_symbol: Some(name),
                ..
            }
            | EventKind::CuptiKernelActivity {
                kernel_name: name, ..
            } => kernel_symbols.push(name.as_str()),
            _ => {}
        }
    }

    Ok((
        NameTable::from_values(requests, u64::MAX as usize)?,
        NameTable::from_values(service_classes, u16::MAX as usize)?,
        NameTable::from_values(kernel_symbols, u32::MAX as usize)?,
    ))
}

fn reserved_references<'a>(capacity: usize) -> Result<Vec<&'a str>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| allocation_error())?;
    Ok(values)
}

fn source_kind(event: &Event) -> Result<hot::SourceKind> {
    match event.source.as_str() {
        "vllm" => Ok(hot::SourceKind::Vllm),
        "host_probe" => Ok(hot::SourceKind::HostProbe),
        "cupti" => Ok(hot::SourceKind::Cupti),
        "device_probe" => Ok(hot::SourceKind::DeviceProbe),
        "policy" => Ok(hot::SourceKind::Policy),
        "workload" => Ok(hot::SourceKind::Workload),
        _ => Err(invalid(event, "unknown event source")),
    }
}

fn parse_pointer(event: &Event, value: &str) -> Result<u64> {
    let digits = value.strip_prefix("0x").unwrap_or(value);
    u64::from_str_radix(digits, 16).map_err(|_| {
        invalid(
            event,
            "kernel_function must be an unsigned hexadecimal address",
        )
    })
}

fn event_id<'a>(input: &'a CompactInput<'_>, sequence: u64) -> &'a str {
    sequence
        .checked_sub(1)
        .and_then(|index| input.sequence_event_ids.get(index as usize).copied())
        .unwrap_or("<synthetic>")
}

fn clock_id(clock_domain: &str) -> u16 {
    match clock_domain {
        "monotonic" => 1,
        "cupti" => 2,
        "realtime" => 3,
        _ => u16::MAX,
    }
}

fn kv_usage_permyriad(value: f64) -> u16 {
    (value.clamp(0.0, 1.0) * 10_000.0).round() as u16
}

fn memcpy_kind(kind: &str) -> u8 {
    match kind {
        "host_to_device" => 1,
        "device_to_host" => 2,
        "device_to_device" => 3,
        "unified" => 4,
        _ => 0,
    }
}

fn invalid(event: &Event, message: &str) -> ObserverError {
    ObserverError::InvalidEvent {
        event_id: event.event_id.clone(),
        message: message.to_owned(),
    }
}

fn allocation_error() -> ObserverError {
    ObserverError::InvalidEvent {
        event_id: "collector".to_owned(),
        message: "memory allocation failed while building the cold compatibility view".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use crate::io::parse_jsonl;

    use super::correlate_checked;

    const MIXED_TRACE: &str = include_str!("../../benchmarks/fixtures/mixed-trace.jsonl");

    #[test]
    fn compatibility_boundary_uses_compact_many_to_many_core() {
        let events = parse_jsonl(MIXED_TRACE).unwrap();
        let report = correlate_checked(&events).unwrap();
        let step = report
            .steps
            .iter()
            .find(|step| step.step_id == 101)
            .unwrap();
        assert_eq!(step.request_slices.len(), 2);
        assert_eq!(step.unmatched_cuda_launches, 0);
        assert!(report.hot_event_record_bytes <= 128);
        assert!(report.core_index_bytes > 0);
    }

    #[test]
    fn cupti_activity_after_step_end_still_matches_submission() {
        let events = parse_jsonl(MIXED_TRACE).unwrap();
        let report = correlate_checked(&events).unwrap();
        let step = report
            .steps
            .iter()
            .find(|step| step.step_id == 101)
            .unwrap();
        assert!(step.kernels.iter().any(|kernel| {
            kernel.gpu_start_timestamp_ns.unwrap() > step.end_timestamp_ns.unwrap()
        }));
    }
}
