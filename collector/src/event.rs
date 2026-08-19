//! Owned JSON compatibility types. These are never used in a probe hot path.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{ObserverError, Result};

pub const CURRENT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Event {
    pub schema_version: u16,
    pub event_id: String,
    pub source: String,
    pub timestamp_ns: u64,
    pub clock_domain: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_timestamp_ns: Option<u64>,
    pub pid: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tid: Option<u32>,
    #[serde(flatten)]
    pub kind: EventKind,
}

impl Event {
    pub fn correlation_timestamp_ns(&self) -> u64 {
        self.normalized_timestamp_ns.unwrap_or(self.timestamp_ns)
    }

    pub fn is_clock_normalized(&self) -> bool {
        self.normalized_timestamp_ns.is_some() || self.clock_domain == "monotonic"
    }

    pub fn validate(&self) -> Result<()> {
        let invalid = |message: &str| ObserverError::InvalidEvent {
            event_id: self.event_id.clone(),
            message: message.to_owned(),
        };

        if self.schema_version != CURRENT_SCHEMA_VERSION {
            return Err(invalid(&format!(
                "unsupported schema version {}; expected {}",
                self.schema_version, CURRENT_SCHEMA_VERSION
            )));
        }
        if self.event_id.trim().is_empty() {
            return Err(invalid("event_id must not be empty"));
        }
        if self.source.trim().is_empty() {
            return Err(invalid("source must not be empty"));
        }
        if self.clock_domain.trim().is_empty() {
            return Err(invalid("clock_domain must not be empty"));
        }

        match &self.kind {
            EventKind::EngineStepBegin {
                request_slices,
                scheduled_tokens,
                prefill_tokens,
                decode_tokens,
                ..
            } => {
                let slice_total: u32 = request_slices
                    .iter()
                    .map(|slice| slice.scheduled_tokens)
                    .sum();
                let phase_total = prefill_tokens.saturating_add(*decode_tokens);
                if slice_total != *scheduled_tokens {
                    return Err(invalid(
                        "sum(request_slices.scheduled_tokens) must equal scheduled_tokens",
                    ));
                }
                if phase_total != *scheduled_tokens {
                    return Err(invalid(
                        "prefill_tokens + decode_tokens must equal scheduled_tokens",
                    ));
                }
            }
            EventKind::CuptiKernelActivity { duration_ns, .. } if *duration_ns == 0 => {
                return Err(invalid("CUPTI kernel duration must be greater than zero"));
            }
            EventKind::CudaKernelLaunch {
                kernel_function, ..
            } if kernel_function.trim().is_empty() => {
                return Err(invalid("kernel_function must not be empty"));
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventKind {
    RequestReceived {
        request_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        service_class: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input_tokens: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        requested_output_tokens: Option<u32>,
    },
    RequestQueued {
        request_id: String,
        queue_depth: u32,
    },
    TokenEmitted {
        request_id: String,
        token_index: u32,
    },
    RequestCompleted {
        request_id: String,
        input_tokens: u32,
        output_tokens: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ttft_ns: Option<u64>,
        e2e_latency_ns: u64,
    },
    EngineStepBegin {
        step_id: u64,
        request_slices: Vec<RequestSlice>,
        prefill_tokens: u32,
        decode_tokens: u32,
        scheduled_tokens: u32,
        queue_depth: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        active_requests: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kv_cache_usage: Option<f64>,
    },
    EngineStepEnd {
        step_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    CudaApiCall {
        function: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ns: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream: Option<u64>,
    },
    CudaKernelLaunch {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<u64>,
        stream: u64,
        kernel_function: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kernel_symbol: Option<String>,
        grid: Dim3,
        block: Dim3,
        shared_memory_bytes: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        graph_id: Option<u64>,
    },
    CudaMemcpy {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream: Option<u64>,
        bytes: u64,
        kind: String,
        asynchronous: bool,
    },
    CudaGraphLaunch {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<u64>,
        stream: u64,
        graph_id: u64,
    },
    CuptiKernelActivity {
        correlation_id: u64,
        stream: u64,
        kernel_name: String,
        duration_ns: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_id: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_id: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        graph_id: Option<u64>,
    },
    MetricSample {
        name: String,
        value: f64,
        unit: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        labels: BTreeMap<String, String>,
    },
    ControlDecision {
        condition: String,
        action: String,
        previous_value: serde_json::Value,
        new_value: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        observed_result: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RequestSlice {
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_id: Option<u64>,
    pub phase: RequestPhase,
    pub scheduled_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_class: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestPhase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
