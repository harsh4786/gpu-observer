#![no_std]

use gpu_observer_core::{Membership, RequestPhase, StepSummary};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorConfig {
    pub large_prefill_tokens: u32,
    pub minimum_gpu_busy_ns: u64,
    pub interactive_service_class_id: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Condition {
    InteractiveDecodeSharesLargePrefill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    LimitBackgroundPrefillAdmission,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Decision {
    pub condition: Condition,
    pub action: Action,
    pub pid: u32,
    pub step_id: u64,
    pub observed_prefill_tokens: u32,
    pub observed_gpu_busy_ns: u64,
}

#[inline]
pub fn detect_prefill_interference(
    config: DetectorConfig,
    step: &StepSummary,
    memberships: &[Membership],
) -> Option<Decision> {
    if step.prefill_tokens < config.large_prefill_tokens
        || step.gpu_busy_time_ns < config.minimum_gpu_busy_ns
    {
        return None;
    }

    let interactive_decode = memberships.iter().any(|membership| {
        membership.slice.phase == RequestPhase::Decode
            && membership.slice.service_class_id == config.interactive_service_class_id
    });
    let background_prefill = memberships.iter().any(|membership| {
        membership.slice.phase == RequestPhase::Prefill
            && membership.slice.service_class_id != config.interactive_service_class_id
    });

    (interactive_decode && background_prefill).then_some(Decision {
        condition: Condition::InteractiveDecodeSharesLargePrefill,
        action: Action::LimitBackgroundPrefillAdmission,
        pid: step.pid,
        step_id: step.step_id,
        observed_prefill_tokens: step.prefill_tokens,
        observed_gpu_busy_ns: step.gpu_busy_time_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_observer_core::StepRequestSliceData;

    fn membership(phase: RequestPhase, service_class_id: u16) -> Membership {
        Membership {
            step_index: 0,
            slice: StepRequestSliceData {
                step_id: 7,
                request_id: service_class_id as u64,
                sequence_id: 1,
                scheduled_tokens: 1,
                service_class_id,
                phase,
                reserved: 0,
            },
        }
    }

    fn step() -> StepSummary {
        StepSummary {
            pid: 42,
            step_id: 7,
            begin_timestamp_ns: 1,
            end_timestamp_ns: 2,
            scheduled_tokens: 257,
            prefill_tokens: 256,
            decode_tokens: 1,
            queue_depth: 3,
            active_requests: 2,
            expected_slices: 2,
            kv_cache_usage_permyriad: 5_000,
            state: 0,
            membership_start: 0,
            membership_count: 2,
            kernel_start: 0,
            kernel_count: 1,
            graph_launch_count: 0,
            unmatched_launch_count: 0,
            kernel_time_sum_ns: 2_000,
            gpu_busy_time_ns: 1_500,
            gpu_span_ns: 1_500,
        }
    }

    #[test]
    fn detects_only_the_interpretable_mixed_step_condition() {
        let config = DetectorConfig {
            large_prefill_tokens: 128,
            minimum_gpu_busy_ns: 1_000,
            interactive_service_class_id: 1,
        };
        let mixed = [
            membership(RequestPhase::Decode, 1),
            membership(RequestPhase::Prefill, 2),
        ];
        assert!(detect_prefill_interference(config, &step(), &mixed).is_some());

        let decode_only = [membership(RequestPhase::Decode, 1)];
        assert_eq!(
            detect_prefill_interference(config, &step(), &decode_only),
            None
        );
    }
}
