//! Emits one synthetic engine step into a semantic ring, spaced out over real
//! wall-clock time, for smoke-testing `semantic_stream_server` (or any other
//! live consumer of the ring) without a running vLLM process.
//!
//! Usage: cargo run --release -p gpu-observer-collector --example live_smoke_emit -- RING_PATH

use std::ffi::CString;
use std::{env, thread, time::Duration};

use gpu_observer_core::RequestPhase;
use gpu_observer_vllm_bridge::{
    gpu_observer_bridge_close, gpu_observer_bridge_open, gpu_observer_bridge_set_focus,
    gpu_observer_emit_output_tokens, gpu_observer_emit_packed_layout,
    gpu_observer_emit_packed_tokens, gpu_observer_emit_step_begin, gpu_observer_emit_step_end,
    OutputTokenInput, PackedSliceInput, PackedTokenInput, SemanticSliceInput,
};

fn now_ns() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    )
    .unwrap()
}

fn main() {
    let path = env::args().nth(1).expect("usage: live_smoke_emit RING_PATH");
    let path_c = CString::new(path).unwrap();
    let bridge = unsafe { gpu_observer_bridge_open(path_c.as_ptr(), 1024) };
    assert!(!bridge.is_null(), "failed to create ring");
    unsafe { gpu_observer_bridge_set_focus(bridge, 0x5EED) };

    if env::var("LIVE_SMOKE_STARTUP_DELAY_MS").is_ok() {
        let delay_ms: u64 = env::var("LIVE_SMOKE_STARTUP_DELAY_MS")
            .unwrap()
            .parse()
            .unwrap();
        eprintln!("live_smoke_emit: ring created, waiting {delay_ms}ms before publishing");
        thread::sleep(Duration::from_millis(delay_ms));
    }

    let request_id = 0x5EED_u64;
    for step_id in 1..=3_u64 {
        let slice = SemanticSliceInput {
            request_id,
            sequence_id: 0,
            scheduled_tokens: 1,
            service_class_id: 0,
            phase: RequestPhase::Decode as u8,
            reserved: 0,
        };
        unsafe {
            gpu_observer_emit_step_begin(bridge, now_ns(), step_id, 1, 0, 1, 0, 1, 100, &slice, 1);
        }
        thread::sleep(Duration::from_millis(80));

        let packed = PackedSliceInput {
            request_id,
            packing_generation: step_id,
            row_begin: 0,
            row_end: 1,
            scheduled_tokens: 1,
            packed_index: 0,
            phase: RequestPhase::Decode as u8,
            reserved: [0; 7],
        };
        unsafe {
            gpu_observer_emit_packed_layout(bridge, now_ns(), step_id, step_id, 1, &packed, 1);
        }
        thread::sleep(Duration::from_millis(40));

        let token = PackedTokenInput {
            request_id,
            packing_generation: step_id,
            packed_row: 0,
            sequence_position: step_id as u32 - 1,
            token_id: 100 + step_id as u32,
            phase: RequestPhase::Decode as u8,
            reserved: [0; 3],
        };
        unsafe {
            gpu_observer_emit_packed_tokens(bridge, now_ns(), step_id, step_id, &token, 1);
        }
        thread::sleep(Duration::from_millis(40));

        let output = OutputTokenInput {
            request_id,
            output_position: step_id as u32 - 1,
            token_id: 100 + step_id as u32,
        };
        unsafe {
            gpu_observer_emit_output_tokens(bridge, now_ns(), step_id, &output, 1);
        }
        thread::sleep(Duration::from_millis(20));

        unsafe {
            gpu_observer_emit_step_end(bridge, now_ns(), step_id, 0);
        }
        eprintln!("live_smoke_emit: published step {step_id}");
        thread::sleep(Duration::from_millis(300));
    }

    unsafe {
        gpu_observer_bridge_close(bridge);
    }
    eprintln!("live_smoke_emit: done");
}
