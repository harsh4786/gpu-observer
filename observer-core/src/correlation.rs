use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::event::{
    CudaKernelLaunchData, CuptiKernelActivityData, EventFlags, EventKind, EventRecord,
    StepRequestSliceData,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorrelationError {
    AllocationFailed,
    IndexOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticCode {
    UnnormalizedClock,
    DuplicateStepBegin,
    StepEndWithoutBegin,
    StepClockReversal,
    OpenStep,
    SliceWithoutOpenStep,
    SliceCountMismatch,
    LaunchOutsideStep,
    ActivityWithoutLaunch,
    TelemetryLoss,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub pid: u32,
    pub sequence: u64,
    pub step_id: u64,
}

pub struct StepState;

impl StepState {
    pub const OPEN: u16 = 1 << 0;
    pub const CLOCK_REVERSAL: u16 = 1 << 1;
    pub const SLICE_COUNT_MISMATCH: u16 = 1 << 2;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepSummary {
    pub pid: u32,
    pub step_id: u64,
    pub begin_timestamp_ns: u64,
    pub end_timestamp_ns: u64,
    pub scheduled_tokens: u32,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub queue_depth: u32,
    pub active_requests: u32,
    pub expected_slices: u32,
    pub kv_cache_usage_permyriad: u16,
    pub state: u16,
    pub membership_start: u32,
    pub membership_count: u32,
    pub kernel_start: u32,
    pub kernel_count: u32,
    pub graph_launch_count: u32,
    pub unmatched_launch_count: u32,
    pub kernel_time_sum_ns: u64,
    pub gpu_busy_time_ns: u64,
    pub gpu_span_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Membership {
    pub step_index: u32,
    pub slice: StepRequestSliceData,
}

pub struct KernelFlags;

impl KernelFlags {
    pub const HAS_ACTIVITY: u16 = 1 << 0;
    pub const HAS_CORRELATION_ID: u16 = 1 << 1;
    pub const HAS_GRAPH_ID: u16 = 1 << 2;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelExecution {
    pub step_index: u32,
    pub pid: u32,
    pub flags: u16,
    pub reserved: u16,
    pub launch_sequence: u64,
    pub activity_sequence: u64,
    pub host_launch_timestamp_ns: u64,
    pub gpu_start_timestamp_ns: u64,
    pub gpu_duration_ns: u64,
    pub launch: CudaKernelLaunchData,
}

pub struct TraceIndex {
    steps: Vec<StepSummary>,
    memberships: Vec<Membership>,
    kernels: Vec<KernelExecution>,
    diagnostics: Vec<Diagnostic>,
}

impl TraceIndex {
    #[inline]
    pub fn steps(&self) -> &[StepSummary] {
        &self.steps
    }

    #[inline]
    pub fn memberships(&self) -> &[Membership] {
        &self.memberships
    }

    #[inline]
    pub fn kernels(&self) -> &[KernelExecution] {
        &self.kernels
    }

    #[inline]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    #[inline]
    pub fn memberships_for_step(&self, step: &StepSummary) -> &[Membership] {
        let start = step.membership_start as usize;
        &self.memberships[start..start + step.membership_count as usize]
    }

    #[inline]
    pub fn kernels_for_step(&self, step: &StepSummary) -> &[KernelExecution] {
        let start = step.kernel_start as usize;
        &self.kernels[start..start + step.kernel_count as usize]
    }

    pub fn allocated_bytes_lower_bound(&self) -> usize {
        self.steps.capacity() * core::mem::size_of::<StepSummary>()
            + self.memberships.capacity() * core::mem::size_of::<Membership>()
            + self.kernels.capacity() * core::mem::size_of::<KernelExecution>()
            + self.diagnostics.capacity() * core::mem::size_of::<Diagnostic>()
    }
}

#[derive(Clone, Copy)]
struct OpenStep {
    pid: u32,
    step_id: u64,
    step_index: u32,
    begin_sequence: u64,
}

#[derive(Clone, Copy)]
struct Activity {
    pid: u32,
    sequence: u64,
    timestamp_ns: u64,
    data: CuptiKernelActivityData,
    matched: bool,
}

pub fn correlate(events: &mut [EventRecord]) -> Result<TraceIndex, CorrelationError> {
    events.sort_unstable_by(event_order);

    let step_count = count(events, EventKind::EngineStepBegin);
    let membership_count = count(events, EventKind::StepRequestSlice);
    let kernel_count = count(events, EventKind::CudaKernelLaunch);
    let activity_count = count(events, EventKind::CuptiKernelActivity);
    if step_count > u32::MAX as usize
        || membership_count > u32::MAX as usize
        || kernel_count > u32::MAX as usize
    {
        return Err(CorrelationError::IndexOverflow);
    }

    let mut steps = reserved_vec(step_count)?;
    let mut memberships = reserved_vec(membership_count)?;
    let mut kernels = reserved_vec(kernel_count)?;
    let mut activities = reserved_vec(activity_count)?;
    let mut open_steps = reserved_vec(step_count.min(16))?;
    let mut diagnostics = reserved_vec(16)?;
    let mut reported_unnormalized_clock = false;

    for event in events.iter() {
        if !reported_unnormalized_clock && !event.header.is_timestamp_normalized() {
            push_diagnostic(
                &mut diagnostics,
                DiagnosticCode::UnnormalizedClock,
                event.header.pid,
                event.header.sequence,
                0,
            )?;
            reported_unnormalized_clock = true;
        }

        if event.header.flags & EventFlags::DROPPED_BEFORE != 0 {
            push_diagnostic(
                &mut diagnostics,
                DiagnosticCode::TelemetryLoss,
                event.header.pid,
                event.header.sequence,
                0,
            )?;
        }

        match event.header.kind {
            EventKind::EngineStepBegin => {
                let data = event.as_engine_step_begin().unwrap();
                if let Some(position) = find_open(&open_steps, event.header.pid, data.step_id) {
                    let previous = open_steps.swap_remove(position);
                    push_diagnostic(
                        &mut diagnostics,
                        DiagnosticCode::DuplicateStepBegin,
                        event.header.pid,
                        previous.begin_sequence,
                        data.step_id,
                    )?;
                }
                let step_index = usize_to_u32(steps.len())?;
                steps.push(StepSummary {
                    pid: event.header.pid,
                    step_id: data.step_id,
                    begin_timestamp_ns: event.header.timestamp_ns,
                    end_timestamp_ns: 0,
                    scheduled_tokens: data.scheduled_tokens,
                    prefill_tokens: data.prefill_tokens,
                    decode_tokens: data.decode_tokens,
                    queue_depth: data.queue_depth,
                    active_requests: data.active_requests,
                    expected_slices: data.expected_slices,
                    kv_cache_usage_permyriad: data.kv_cache_usage_permyriad,
                    state: StepState::OPEN,
                    membership_start: 0,
                    membership_count: 0,
                    kernel_start: 0,
                    kernel_count: 0,
                    graph_launch_count: 0,
                    unmatched_launch_count: 0,
                    kernel_time_sum_ns: 0,
                    gpu_busy_time_ns: 0,
                    gpu_span_ns: 0,
                });
                push_open(
                    &mut open_steps,
                    OpenStep {
                        pid: event.header.pid,
                        step_id: data.step_id,
                        step_index,
                        begin_sequence: event.header.sequence,
                    },
                )?;
            }
            EventKind::StepRequestSlice => {
                let slice = event.as_step_request_slice().unwrap();
                if let Some(position) = find_open(&open_steps, event.header.pid, slice.step_id) {
                    memberships.push(Membership {
                        step_index: open_steps[position].step_index,
                        slice,
                    });
                } else {
                    push_diagnostic(
                        &mut diagnostics,
                        DiagnosticCode::SliceWithoutOpenStep,
                        event.header.pid,
                        event.header.sequence,
                        slice.step_id,
                    )?;
                }
            }
            EventKind::EngineStepEnd => {
                let data = event.as_engine_step_end().unwrap();
                if let Some(position) = find_open(&open_steps, event.header.pid, data.step_id) {
                    let open = open_steps.swap_remove(position);
                    let step = &mut steps[open.step_index as usize];
                    step.state &= !StepState::OPEN;
                    step.end_timestamp_ns = event.header.timestamp_ns;
                    if step.end_timestamp_ns < step.begin_timestamp_ns {
                        step.state |= StepState::CLOCK_REVERSAL;
                        push_diagnostic(
                            &mut diagnostics,
                            DiagnosticCode::StepClockReversal,
                            event.header.pid,
                            event.header.sequence,
                            data.step_id,
                        )?;
                    }
                } else {
                    push_diagnostic(
                        &mut diagnostics,
                        DiagnosticCode::StepEndWithoutBegin,
                        event.header.pid,
                        event.header.sequence,
                        data.step_id,
                    )?;
                }
            }
            EventKind::CudaKernelLaunch => {
                let launch = event.as_cuda_kernel_launch().unwrap();
                if let Some(open) = most_recent_open_for_pid(&open_steps, event.header.pid) {
                    let mut flags = 0;
                    if event.header.has_correlation_id() {
                        flags |= KernelFlags::HAS_CORRELATION_ID;
                    }
                    if event.header.has_graph_id() {
                        flags |= KernelFlags::HAS_GRAPH_ID;
                    }
                    kernels.push(KernelExecution {
                        step_index: open.step_index,
                        pid: event.header.pid,
                        flags,
                        reserved: 0,
                        launch_sequence: event.header.sequence,
                        activity_sequence: 0,
                        host_launch_timestamp_ns: event.header.timestamp_ns,
                        gpu_start_timestamp_ns: 0,
                        gpu_duration_ns: 0,
                        launch,
                    });
                } else {
                    push_diagnostic(
                        &mut diagnostics,
                        DiagnosticCode::LaunchOutsideStep,
                        event.header.pid,
                        event.header.sequence,
                        0,
                    )?;
                }
            }
            EventKind::CudaGraphLaunch => {
                if let Some(open) = most_recent_open_for_pid(&open_steps, event.header.pid) {
                    steps[open.step_index as usize].graph_launch_count += 1;
                } else {
                    push_diagnostic(
                        &mut diagnostics,
                        DiagnosticCode::LaunchOutsideStep,
                        event.header.pid,
                        event.header.sequence,
                        0,
                    )?;
                }
            }
            EventKind::CuptiKernelActivity => {
                activities.push(Activity {
                    pid: event.header.pid,
                    sequence: event.header.sequence,
                    timestamp_ns: event.header.timestamp_ns,
                    data: event.as_cupti_kernel_activity().unwrap(),
                    matched: false,
                });
            }
            _ => {}
        }
    }

    for open in open_steps {
        push_diagnostic(
            &mut diagnostics,
            DiagnosticCode::OpenStep,
            open.pid,
            open.begin_sequence,
            open.step_id,
        )?;
    }

    memberships.sort_unstable_by_key(|membership| {
        (
            membership.step_index,
            membership.slice.request_id,
            membership.slice.sequence_id,
        )
    });
    assign_membership_ranges(&mut steps, &memberships, &mut diagnostics)?;

    activities.sort_unstable_by(activity_order);
    match_activities(&mut kernels, &mut activities);
    for activity in activities.iter().filter(|activity| !activity.matched) {
        push_diagnostic(
            &mut diagnostics,
            DiagnosticCode::ActivityWithoutLaunch,
            activity.pid,
            activity.sequence,
            0,
        )?;
    }

    kernels.sort_unstable_by_key(|kernel| {
        (
            kernel.step_index,
            if kernel.flags & KernelFlags::HAS_ACTIVITY != 0 {
                kernel.gpu_start_timestamp_ns
            } else {
                u64::MAX
            },
            kernel.launch_sequence,
        )
    });
    assign_kernel_ranges_and_timing(&mut steps, &kernels)?;

    Ok(TraceIndex {
        steps,
        memberships,
        kernels,
        diagnostics,
    })
}

fn count(events: &[EventRecord], kind: EventKind) -> usize {
    events
        .iter()
        .filter(|event| event.header.kind == kind)
        .count()
}

fn reserved_vec<T>(capacity: usize) -> Result<Vec<T>, CorrelationError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| CorrelationError::AllocationFailed)?;
    Ok(values)
}

fn push_open(open_steps: &mut Vec<OpenStep>, step: OpenStep) -> Result<(), CorrelationError> {
    if open_steps.len() == open_steps.capacity() {
        open_steps
            .try_reserve(16)
            .map_err(|_| CorrelationError::AllocationFailed)?;
    }
    open_steps.push(step);
    Ok(())
}

fn push_diagnostic(
    diagnostics: &mut Vec<Diagnostic>,
    code: DiagnosticCode,
    pid: u32,
    sequence: u64,
    step_id: u64,
) -> Result<(), CorrelationError> {
    if diagnostics.len() == diagnostics.capacity() {
        diagnostics
            .try_reserve(16)
            .map_err(|_| CorrelationError::AllocationFailed)?;
    }
    diagnostics.push(Diagnostic {
        code,
        pid,
        sequence,
        step_id,
    });
    Ok(())
}

fn usize_to_u32(value: usize) -> Result<u32, CorrelationError> {
    u32::try_from(value).map_err(|_| CorrelationError::IndexOverflow)
}

fn event_order(left: &EventRecord, right: &EventRecord) -> Ordering {
    (
        left.header.timestamp_ns,
        left.header.pid,
        kind_rank(left.header.kind),
        left.header.sequence,
    )
        .cmp(&(
            right.header.timestamp_ns,
            right.header.pid,
            kind_rank(right.header.kind),
            right.header.sequence,
        ))
}

const fn kind_rank(kind: EventKind) -> u8 {
    match kind {
        EventKind::EngineStepBegin => 0,
        EventKind::StepRequestSlice => 1,
        EventKind::CudaKernelLaunch | EventKind::CudaGraphLaunch | EventKind::CudaMemcpy => 2,
        EventKind::EngineStepEnd => 3,
        _ => 4,
    }
}

fn find_open(open: &[OpenStep], pid: u32, step_id: u64) -> Option<usize> {
    open.iter()
        .rposition(|step| step.pid == pid && step.step_id == step_id)
}

fn most_recent_open_for_pid(open: &[OpenStep], pid: u32) -> Option<OpenStep> {
    open.iter().rfind(|step| step.pid == pid).copied()
}

fn assign_membership_ranges(
    steps: &mut [StepSummary],
    memberships: &[Membership],
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<(), CorrelationError> {
    let mut cursor = 0;
    for (step_index, step) in steps.iter_mut().enumerate() {
        let start = cursor;
        while cursor < memberships.len() && memberships[cursor].step_index as usize == step_index {
            cursor += 1;
        }
        step.membership_start = usize_to_u32(start)?;
        step.membership_count = usize_to_u32(cursor - start)?;
        if step.membership_count != step.expected_slices {
            step.state |= StepState::SLICE_COUNT_MISMATCH;
            push_diagnostic(
                diagnostics,
                DiagnosticCode::SliceCountMismatch,
                step.pid,
                0,
                step.step_id,
            )?;
        }
    }
    Ok(())
}

fn activity_order(left: &Activity, right: &Activity) -> Ordering {
    activity_key(left)
        .cmp(&activity_key(right))
        .then_with(|| left.timestamp_ns.cmp(&right.timestamp_ns))
        .then_with(|| left.sequence.cmp(&right.sequence))
}

fn activity_key(activity: &Activity) -> (u32, u64, u64) {
    (
        activity.pid,
        activity.data.correlation_id,
        activity.data.stream,
    )
}

fn match_activities(kernels: &mut [KernelExecution], activities: &mut [Activity]) {
    for kernel in kernels {
        if kernel.flags & KernelFlags::HAS_CORRELATION_ID == 0 {
            continue;
        }
        let key = (
            kernel.pid,
            kernel.launch.correlation_id,
            kernel.launch.stream,
        );
        let start = activities.partition_point(|activity| activity_key(activity) < key);
        let mut best = None;
        for (offset, activity) in activities[start..].iter().enumerate() {
            if activity_key(activity) != key {
                break;
            }
            if !activity.matched {
                best = Some(start + offset);
                if activity.timestamp_ns >= kernel.host_launch_timestamp_ns {
                    break;
                }
            }
        }
        if let Some(index) = best {
            let activity = &mut activities[index];
            activity.matched = true;
            kernel.flags |= KernelFlags::HAS_ACTIVITY;
            kernel.activity_sequence = activity.sequence;
            kernel.gpu_start_timestamp_ns = activity.timestamp_ns;
            kernel.gpu_duration_ns = activity.data.duration_ns;
            if kernel.launch.kernel_symbol_id == 0 {
                kernel.launch.kernel_symbol_id = activity.data.kernel_symbol_id;
            }
        }
    }
}

fn assign_kernel_ranges_and_timing(
    steps: &mut [StepSummary],
    kernels: &[KernelExecution],
) -> Result<(), CorrelationError> {
    let mut cursor = 0;
    for (step_index, step) in steps.iter_mut().enumerate() {
        let start = cursor;
        while cursor < kernels.len() && kernels[cursor].step_index as usize == step_index {
            cursor += 1;
        }
        step.kernel_start = usize_to_u32(start)?;
        step.kernel_count = usize_to_u32(cursor - start)?;
        let step_kernels = &kernels[start..cursor];
        step.unmatched_launch_count = usize_to_u32(
            step_kernels
                .iter()
                .filter(|kernel| kernel.flags & KernelFlags::HAS_ACTIVITY == 0)
                .count(),
        )?;
        step.kernel_time_sum_ns = step_kernels
            .iter()
            .map(|kernel| kernel.gpu_duration_ns)
            .sum();

        let mut first = None;
        let mut union_start = 0;
        let mut union_end = 0;
        let mut busy = 0_u64;
        for kernel in step_kernels
            .iter()
            .filter(|kernel| kernel.flags & KernelFlags::HAS_ACTIVITY != 0)
        {
            let start_ns = kernel.gpu_start_timestamp_ns;
            let end_ns = start_ns.saturating_add(kernel.gpu_duration_ns);
            if first.is_none() {
                first = Some(start_ns);
                union_start = start_ns;
                union_end = end_ns;
            } else if start_ns <= union_end {
                union_end = union_end.max(end_ns);
            } else {
                busy = busy.saturating_add(union_end.saturating_sub(union_start));
                union_start = start_ns;
                union_end = end_ns;
            }
        }
        if let Some(first_ns) = first {
            busy = busy.saturating_add(union_end.saturating_sub(union_start));
            step.gpu_busy_time_ns = busy;
            step.gpu_span_ns = union_end.saturating_sub(first_ns);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::mem::size_of;

    use super::*;
    use crate::event::{
        Dim3, EngineStepBeginData, EngineStepEndData, EventContext, RequestPhase, SourceKind,
    };

    fn context(source: SourceKind, timestamp_ns: u64, sequence: u64, flags: u16) -> EventContext {
        EventContext {
            source_timestamp_ns: timestamp_ns,
            timestamp_ns,
            sequence,
            pid: 42,
            tid: 43,
            clock_id: 1,
            flags: flags | EventFlags::TIMESTAMP_NORMALIZED,
            source,
        }
    }

    #[test]
    fn mixed_step_stays_many_to_many_and_links_async_gpu_activity() {
        let mut events = vec![
            EventRecord::engine_step_begin(
                context(SourceKind::Vllm, 100, 1, 0),
                EngineStepBeginData {
                    step_id: 172,
                    scheduled_tokens: 257,
                    prefill_tokens: 256,
                    decode_tokens: 1,
                    queue_depth: 3,
                    active_requests: 2,
                    expected_slices: 2,
                    kv_cache_usage_permyriad: 5_000,
                    reserved: 0,
                },
            ),
            EventRecord::step_request_slice(
                context(SourceKind::Vllm, 101, 2, 0),
                StepRequestSliceData {
                    step_id: 172,
                    request_id: 1,
                    sequence_id: 1,
                    scheduled_tokens: 1,
                    service_class_id: 1,
                    phase: RequestPhase::Decode,
                    reserved: 0,
                },
            ),
            EventRecord::step_request_slice(
                context(SourceKind::Vllm, 102, 3, 0),
                StepRequestSliceData {
                    step_id: 172,
                    request_id: 2,
                    sequence_id: 2,
                    scheduled_tokens: 256,
                    service_class_id: 2,
                    phase: RequestPhase::Prefill,
                    reserved: 0,
                },
            ),
            EventRecord::cuda_kernel_launch(
                context(
                    SourceKind::HostProbe,
                    105,
                    4,
                    EventFlags::HAS_CORRELATION_ID,
                ),
                CudaKernelLaunchData {
                    correlation_id: 99,
                    stream: 7,
                    kernel_function: 0x1234,
                    graph_id: 0,
                    grid: Dim3 { x: 8, y: 1, z: 1 },
                    block: Dim3 { x: 128, y: 1, z: 1 },
                    shared_memory_bytes: 0,
                    kernel_symbol_id: 10,
                },
            ),
            EventRecord::engine_step_end(
                context(SourceKind::Vllm, 110, 5, 0),
                EngineStepEndData {
                    step_id: 172,
                    status: 0,
                    reserved: [0; 7],
                },
            ),
            EventRecord::cupti_kernel_activity(
                context(SourceKind::Cupti, 120, 6, 0),
                CuptiKernelActivityData {
                    correlation_id: 99,
                    stream: 7,
                    duration_ns: 20,
                    graph_id: 0,
                    kernel_symbol_id: 10,
                    device_id: 0,
                    context_id: 1,
                    reserved: 0,
                },
            ),
        ];

        let trace = correlate(&mut events).unwrap();
        assert_eq!(trace.steps().len(), 1);
        assert_eq!(trace.memberships_for_step(&trace.steps()[0]).len(), 2);
        assert_eq!(trace.kernels_for_step(&trace.steps()[0]).len(), 1);
        assert_eq!(trace.steps()[0].gpu_busy_time_ns, 20);
        assert_eq!(trace.kernels()[0].gpu_start_timestamp_ns, 120);
    }

    #[test]
    fn loss_flag_downgrades_trace_precision() {
        let mut events = [EventRecord::engine_step_end(
            context(SourceKind::Vllm, 100, 4, EventFlags::DROPPED_BEFORE),
            EngineStepEndData {
                step_id: 172,
                status: 0,
                reserved: [0; 7],
            },
        )];
        let trace = correlate(&mut events).unwrap();
        assert!(trace
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::TelemetryLoss));
    }

    #[test]
    fn hot_index_records_have_bounded_layouts() {
        assert!(size_of::<StepSummary>() <= 112);
        assert!(size_of::<Membership>() <= 40);
        assert!(size_of::<KernelExecution>() <= 128);
    }
}
