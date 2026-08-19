//! Owned report types for storage and UI consumption outside inference.

use serde::{Deserialize, Serialize};

use crate::event::{Dim3, RequestSlice, CURRENT_SCHEMA_VERSION};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CorrelationReport {
    pub schema_version: u16,
    pub input_event_count: usize,
    pub hot_event_record_bytes: usize,
    pub core_index_bytes: usize,
    pub steps: Vec<CorrelatedStep>,
    pub requests: Vec<RequestExecution>,
    pub unassigned_events: Vec<UnassignedEvent>,
    pub diagnostics: Vec<Diagnostic>,
}

impl CorrelationReport {
    pub fn empty(input_event_count: usize) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            input_event_count,
            hot_event_record_bytes: gpu_observer_core::EVENT_RECORD_BYTES,
            core_index_bytes: 0,
            steps: Vec::new(),
            requests: Vec::new(),
            unassigned_events: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CorrelatedStep {
    pub step_id: u64,
    pub pid: u32,
    pub begin_timestamp_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_timestamp_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_time_ns: Option<u64>,
    pub request_slices: Vec<RequestSlice>,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub scheduled_tokens: u32,
    pub queue_depth: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_requests: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_usage: Option<f64>,
    pub cuda_launch_count: usize,
    pub graph_launch_count: usize,
    pub kernels: Vec<KernelExecution>,
    pub unmatched_cuda_launches: usize,
    pub kernel_time_sum_ns: u64,
    pub gpu_busy_time_ns: u64,
    pub gpu_span_ns: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct KernelExecution {
    pub launch_event_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activity_event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<u64>,
    pub stream: u64,
    pub kernel_function: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_name: Option<String>,
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_memory_bytes: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<u64>,
    pub host_launch_timestamp_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_start_timestamp_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_duration_ns: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RequestExecution {
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_class: Option<String>,
    pub step_ids: Vec<u64>,
    pub shared_with_request_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_delay_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e2e_latency_ns: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UnassignedEvent {
    pub event_id: String,
    pub event: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}
