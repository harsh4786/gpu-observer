use core::fmt::{Debug, Formatter};
use core::mem::size_of;

pub const SCHEMA_VERSION: u16 = 1;
pub const EVENT_PAYLOAD_BYTES: usize = 64;
pub const EVENT_RECORD_BYTES: usize = 104;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceKind {
    Vllm = 1,
    HostProbe = 2,
    Cupti = 3,
    DeviceProbe = 4,
    Policy = 5,
    Workload = 6,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind {
    RequestReceived = 1,
    RequestQueued = 2,
    TokenEmitted = 3,
    RequestCompleted = 4,
    EngineStepBegin = 10,
    StepRequestSlice = 11,
    EngineStepEnd = 12,
    PackedLayoutBegin = 13,
    PackedRequestSlice = 14,
    PackedTokenRow = 15,
    AcceptedOutputToken = 16,
    CudaKernelLaunch = 20,
    CuptiKernelActivity = 21,
    CudaGraphLaunch = 22,
    CudaMemcpy = 23,
    MetricSample = 30,
    ControlDecision = 40,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestPhase {
    Prefill = 1,
    Decode = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub struct EventFlags;

impl EventFlags {
    pub const TIMESTAMP_NORMALIZED: u16 = 1 << 0;
    pub const HAS_CORRELATION_ID: u16 = 1 << 1;
    pub const HAS_GRAPH_ID: u16 = 1 << 2;
    pub const HAS_TTFT: u16 = 1 << 3;
    pub const HAS_KV_CACHE_USAGE: u16 = 1 << 4;
    pub const DROPPED_BEFORE: u16 = 1 << 5;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventHeader {
    pub source_timestamp_ns: u64,
    pub timestamp_ns: u64,
    pub sequence: u64,
    pub pid: u32,
    pub tid: u32,
    pub clock_id: u16,
    pub schema_version: u16,
    pub flags: u16,
    pub source: SourceKind,
    pub kind: EventKind,
}

impl EventHeader {
    #[inline]
    pub const fn is_timestamp_normalized(&self) -> bool {
        self.flags & EventFlags::TIMESTAMP_NORMALIZED != 0
    }

    #[inline]
    pub const fn has_correlation_id(&self) -> bool {
        self.flags & EventFlags::HAS_CORRELATION_ID != 0
    }

    #[inline]
    pub const fn has_graph_id(&self) -> bool {
        self.flags & EventFlags::HAS_GRAPH_ID != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventContext {
    pub source_timestamp_ns: u64,
    pub timestamp_ns: u64,
    pub sequence: u64,
    pub pid: u32,
    pub tid: u32,
    pub clock_id: u16,
    pub flags: u16,
    pub source: SourceKind,
}

impl EventContext {
    #[inline]
    pub const fn header(self, kind: EventKind) -> EventHeader {
        EventHeader {
            source_timestamp_ns: self.source_timestamp_ns,
            timestamp_ns: self.timestamp_ns,
            sequence: self.sequence,
            pid: self.pid,
            tid: self.tid,
            clock_id: self.clock_id,
            schema_version: SCHEMA_VERSION,
            flags: self.flags,
            source: self.source,
            kind,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestReceivedData {
    pub request_id: u64,
    pub input_tokens: u32,
    pub requested_output_tokens: u32,
    pub service_class_id: u16,
    pub reserved: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestQueuedData {
    pub request_id: u64,
    pub queue_depth: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenEmittedData {
    pub request_id: u64,
    pub token_index: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestCompletedData {
    pub request_id: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub ttft_ns: u64,
    pub e2e_latency_ns: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineStepBeginData {
    pub step_id: u64,
    pub scheduled_tokens: u32,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub queue_depth: u32,
    pub active_requests: u32,
    pub expected_slices: u32,
    pub kv_cache_usage_permyriad: u16,
    pub reserved: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepRequestSliceData {
    pub step_id: u64,
    pub request_id: u64,
    pub sequence_id: u64,
    pub scheduled_tokens: u32,
    pub service_class_id: u16,
    pub phase: RequestPhase,
    pub reserved: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineStepEndData {
    pub step_id: u64,
    pub status: u8,
    pub reserved: [u8; 7],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedLayoutBeginData {
    pub step_id: u64,
    pub packing_generation: u64,
    pub total_tokens: u32,
    pub expected_slices: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedRequestSliceData {
    pub step_id: u64,
    pub request_id: u64,
    pub packing_generation: u64,
    pub row_begin: u32,
    pub row_end: u32,
    pub scheduled_tokens: u32,
    pub packed_index: u32,
    pub phase: RequestPhase,
    pub reserved: [u8; 7],
}

/// One authoritative model-input token after GPUModelRunner packing.
///
/// Emission is opt-in for a single focused request. Keeping this flat avoids
/// variable-length payloads in the inference-process ring.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedTokenRowData {
    pub step_id: u64,
    pub request_id: u64,
    pub packing_generation: u64,
    pub packed_row: u32,
    pub sequence_position: u32,
    pub token_id: u32,
    pub phase: RequestPhase,
    pub reserved: [u8; 3],
}

/// One token accepted by the scheduler after model execution.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptedOutputTokenData {
    pub step_id: u64,
    pub request_id: u64,
    pub output_position: u32,
    pub token_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKernelLaunchData {
    pub correlation_id: u64,
    pub stream: u64,
    pub kernel_function: u64,
    pub graph_id: u64,
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_memory_bytes: u32,
    pub kernel_symbol_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CuptiKernelActivityData {
    pub correlation_id: u64,
    pub stream: u64,
    pub duration_ns: u64,
    pub graph_id: u64,
    pub kernel_symbol_id: u32,
    pub device_id: u32,
    pub context_id: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaGraphLaunchData {
    pub correlation_id: u64,
    pub stream: u64,
    pub graph_id: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaMemcpyData {
    pub correlation_id: u64,
    pub stream: u64,
    pub bytes: u64,
    pub kind: u8,
    pub asynchronous: u8,
    pub reserved: [u8; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricSampleData {
    pub value: f64,
    pub series_id: u32,
    pub unit_id: u16,
    pub reserved: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControlDecisionData {
    pub previous_value: i64,
    pub new_value: i64,
    pub condition_id: u32,
    pub action_id: u32,
    pub parameter_id: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union EventData {
    request_received: RequestReceivedData,
    request_queued: RequestQueuedData,
    token_emitted: TokenEmittedData,
    request_completed: RequestCompletedData,
    engine_step_begin: EngineStepBeginData,
    step_request_slice: StepRequestSliceData,
    engine_step_end: EngineStepEndData,
    packed_layout_begin: PackedLayoutBeginData,
    packed_request_slice: PackedRequestSliceData,
    packed_token_row: PackedTokenRowData,
    accepted_output_token: AcceptedOutputTokenData,
    cuda_kernel_launch: CudaKernelLaunchData,
    cupti_kernel_activity: CuptiKernelActivityData,
    cuda_graph_launch: CudaGraphLaunchData,
    cuda_memcpy: CudaMemcpyData,
    metric_sample: MetricSampleData,
    control_decision: ControlDecisionData,
    raw: [u8; EVENT_PAYLOAD_BYTES],
}

#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct EventRecord {
    pub header: EventHeader,
    data: EventData,
}

macro_rules! event_constructor {
    ($name:ident, $kind:ident, $field:ident, $data:ty) => {
        #[inline]
        pub const fn $name(context: EventContext, data: $data) -> Self {
            Self {
                header: context.header(EventKind::$kind),
                data: EventData { $field: data },
            }
        }
    };
}

macro_rules! event_accessor {
    ($name:ident, $kind:ident, $field:ident, $data:ty) => {
        #[inline]
        pub fn $name(&self) -> Option<$data> {
            if self.header.kind == EventKind::$kind {
                Some(unsafe { self.data.$field })
            } else {
                None
            }
        }
    };
}

impl EventRecord {
    event_constructor!(
        request_received,
        RequestReceived,
        request_received,
        RequestReceivedData
    );
    event_constructor!(
        request_queued,
        RequestQueued,
        request_queued,
        RequestQueuedData
    );
    event_constructor!(token_emitted, TokenEmitted, token_emitted, TokenEmittedData);
    event_constructor!(
        request_completed,
        RequestCompleted,
        request_completed,
        RequestCompletedData
    );
    event_constructor!(
        engine_step_begin,
        EngineStepBegin,
        engine_step_begin,
        EngineStepBeginData
    );
    event_constructor!(
        step_request_slice,
        StepRequestSlice,
        step_request_slice,
        StepRequestSliceData
    );
    event_constructor!(
        engine_step_end,
        EngineStepEnd,
        engine_step_end,
        EngineStepEndData
    );
    event_constructor!(
        packed_layout_begin,
        PackedLayoutBegin,
        packed_layout_begin,
        PackedLayoutBeginData
    );
    event_constructor!(
        packed_request_slice,
        PackedRequestSlice,
        packed_request_slice,
        PackedRequestSliceData
    );
    event_constructor!(
        packed_token_row,
        PackedTokenRow,
        packed_token_row,
        PackedTokenRowData
    );
    event_constructor!(
        accepted_output_token,
        AcceptedOutputToken,
        accepted_output_token,
        AcceptedOutputTokenData
    );
    event_constructor!(
        cuda_kernel_launch,
        CudaKernelLaunch,
        cuda_kernel_launch,
        CudaKernelLaunchData
    );
    event_constructor!(
        cupti_kernel_activity,
        CuptiKernelActivity,
        cupti_kernel_activity,
        CuptiKernelActivityData
    );
    event_constructor!(
        cuda_graph_launch,
        CudaGraphLaunch,
        cuda_graph_launch,
        CudaGraphLaunchData
    );
    event_constructor!(cuda_memcpy, CudaMemcpy, cuda_memcpy, CudaMemcpyData);
    event_constructor!(metric_sample, MetricSample, metric_sample, MetricSampleData);
    event_constructor!(
        control_decision,
        ControlDecision,
        control_decision,
        ControlDecisionData
    );

    event_accessor!(
        as_request_received,
        RequestReceived,
        request_received,
        RequestReceivedData
    );
    event_accessor!(
        as_request_queued,
        RequestQueued,
        request_queued,
        RequestQueuedData
    );
    event_accessor!(
        as_token_emitted,
        TokenEmitted,
        token_emitted,
        TokenEmittedData
    );
    event_accessor!(
        as_request_completed,
        RequestCompleted,
        request_completed,
        RequestCompletedData
    );
    event_accessor!(
        as_engine_step_begin,
        EngineStepBegin,
        engine_step_begin,
        EngineStepBeginData
    );
    event_accessor!(
        as_step_request_slice,
        StepRequestSlice,
        step_request_slice,
        StepRequestSliceData
    );
    event_accessor!(
        as_engine_step_end,
        EngineStepEnd,
        engine_step_end,
        EngineStepEndData
    );
    event_accessor!(
        as_packed_layout_begin,
        PackedLayoutBegin,
        packed_layout_begin,
        PackedLayoutBeginData
    );
    event_accessor!(
        as_packed_request_slice,
        PackedRequestSlice,
        packed_request_slice,
        PackedRequestSliceData
    );
    event_accessor!(
        as_packed_token_row,
        PackedTokenRow,
        packed_token_row,
        PackedTokenRowData
    );
    event_accessor!(
        as_accepted_output_token,
        AcceptedOutputToken,
        accepted_output_token,
        AcceptedOutputTokenData
    );
    event_accessor!(
        as_cuda_kernel_launch,
        CudaKernelLaunch,
        cuda_kernel_launch,
        CudaKernelLaunchData
    );
    event_accessor!(
        as_cupti_kernel_activity,
        CuptiKernelActivity,
        cupti_kernel_activity,
        CuptiKernelActivityData
    );
    event_accessor!(
        as_cuda_graph_launch,
        CudaGraphLaunch,
        cuda_graph_launch,
        CudaGraphLaunchData
    );
    event_accessor!(as_cuda_memcpy, CudaMemcpy, cuda_memcpy, CudaMemcpyData);
    event_accessor!(
        as_metric_sample,
        MetricSample,
        metric_sample,
        MetricSampleData
    );
    event_accessor!(
        as_control_decision,
        ControlDecision,
        control_decision,
        ControlDecisionData
    );
}

impl Debug for EventRecord {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("EventRecord")
            .field("header", &self.header)
            .finish_non_exhaustive()
    }
}

const _: () = assert!(size_of::<EventHeader>() == 40);
const _: () = assert!(size_of::<EventData>() == EVENT_PAYLOAD_BYTES);
const _: () = assert!(size_of::<CudaKernelLaunchData>() == EVENT_PAYLOAD_BYTES);
const _: () = assert!(size_of::<EventRecord>() == EVENT_RECORD_BYTES);

#[cfg(test)]
mod tests {
    use super::*;

    const CONTEXT: EventContext = EventContext {
        source_timestamp_ns: 100,
        timestamp_ns: 110,
        sequence: 7,
        pid: 42,
        tid: 43,
        clock_id: 1,
        flags: EventFlags::TIMESTAMP_NORMALIZED,
        source: SourceKind::Vllm,
    };

    #[test]
    fn record_layout_is_fixed_and_dense() {
        assert_eq!(size_of::<EventRecord>(), EVENT_RECORD_BYTES);
        assert_eq!(core::mem::align_of::<EventRecord>(), 8);
    }

    #[test]
    fn typed_access_does_not_allocate_or_cast_invalid_payloads() {
        let data = EngineStepBeginData {
            step_id: 172,
            scheduled_tokens: 258,
            prefill_tokens: 256,
            decode_tokens: 2,
            queue_depth: 18,
            active_requests: 3,
            expected_slices: 3,
            kv_cache_usage_permyriad: 7_500,
            reserved: 0,
        };
        let event = EventRecord::engine_step_begin(CONTEXT, data);
        assert_eq!(event.as_engine_step_begin(), Some(data));
        assert_eq!(event.as_cuda_kernel_launch(), None);
    }
}
