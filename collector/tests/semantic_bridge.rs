use std::ffi::CString;
use std::fs;

use gpu_observer_collector::semantic::SemanticRingReader;
use gpu_observer_core::{RequestPhase, SemanticRecordKind};
use gpu_observer_vllm_bridge::{
    gpu_observer_bridge_close, gpu_observer_bridge_focus_close, gpu_observer_bridge_focus_open,
    gpu_observer_bridge_focus_set, gpu_observer_bridge_get_focus, gpu_observer_bridge_open,
    gpu_observer_bridge_set_focus, gpu_observer_emit_output_tokens, gpu_observer_emit_packed_layout,
    gpu_observer_emit_packed_tokens, gpu_observer_emit_step_begin, gpu_observer_emit_step_end,
    OutputTokenInput, PackedSliceInput, PackedTokenInput, SemanticSliceInput,
};

#[test]
fn native_producer_and_collector_share_fixed_records() {
    let path = std::env::temp_dir().join(format!(
        "gpu-observer-semantic-test-{}-{}.ring",
        std::process::id(),
        std::thread::current().name().unwrap_or("unnamed")
    ));
    let path_string = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let bridge = unsafe { gpu_observer_bridge_open(path_string.as_ptr(), 64) };
    assert!(!bridge.is_null());

    let input = SemanticSliceInput {
        request_id: 0x1234,
        sequence_id: 0,
        scheduled_tokens: 44,
        service_class_id: 0,
        phase: RequestPhase::Prefill as u8,
        reserved: 0,
    };
    assert_eq!(
        unsafe { gpu_observer_emit_step_begin(bridge, 100, 1, 44, 44, 0, 3, 4, 125, &input, 1,) },
        0
    );
    let packed = PackedSliceInput {
        request_id: 0x1234,
        packing_generation: 9,
        row_begin: 0,
        row_end: 44,
        scheduled_tokens: 44,
        packed_index: 0,
        phase: RequestPhase::Prefill as u8,
        reserved: [0; 7],
    };
    assert_eq!(
        unsafe { gpu_observer_emit_packed_layout(bridge, 150, 1, 9, 44, &packed, 1) },
        0
    );
    let packed_token = PackedTokenInput {
        request_id: 0x1234,
        packing_generation: 9,
        packed_row: 0,
        sequence_position: 17,
        token_id: 9707,
        phase: RequestPhase::Prefill as u8,
        reserved: [0; 3],
    };
    assert_eq!(
        unsafe { gpu_observer_emit_packed_tokens(bridge, 160, 1, 9, &packed_token, 1) },
        0
    );
    let output_token = OutputTokenInput {
        request_id: 0x1234,
        output_position: 0,
        token_id: 3838,
    };
    assert_eq!(
        unsafe { gpu_observer_emit_output_tokens(bridge, 190, 1, &output_token, 1) },
        0
    );
    assert_eq!(unsafe { gpu_observer_emit_step_end(bridge, 200, 1, 0) }, 0);

    let mut reader = SemanticRingReader::open(&path).unwrap();
    let begin = reader.try_next().unwrap().unwrap();
    let slice = reader.try_next().unwrap().unwrap();
    let packed_begin = reader.try_next().unwrap().unwrap();
    let packed_slice = reader.try_next().unwrap().unwrap();
    let packed_token = reader.try_next().unwrap().unwrap();
    let output_token = reader.try_next().unwrap().unwrap();
    let end = reader.try_next().unwrap().unwrap();
    assert_eq!(begin.kind, SemanticRecordKind::ENGINE_STEP_BEGIN);
    assert_eq!(begin.prefill_tokens, 44);
    assert_eq!(slice.kind, SemanticRecordKind::STEP_REQUEST_SLICE);
    assert_eq!(slice.request_id, 0x1234);
    assert_eq!(packed_begin.kind, SemanticRecordKind::PACKED_LAYOUT_BEGIN);
    assert_eq!(packed_begin.sequence_id, 9);
    assert_eq!(packed_slice.kind, SemanticRecordKind::PACKED_REQUEST_SLICE);
    assert_eq!(packed_slice.request_id, 0x1234);
    assert_eq!(packed_slice.prefill_tokens, 0);
    assert_eq!(packed_slice.decode_tokens, 44);
    assert_eq!(packed_token.kind, SemanticRecordKind::PACKED_TOKEN_ROW);
    assert_eq!(packed_token.request_id, 0x1234);
    assert_eq!(packed_token.prefill_tokens, 0);
    assert_eq!(packed_token.decode_tokens, 17);
    assert_eq!(packed_token.scheduled_tokens, 9707);
    assert_eq!(output_token.kind, SemanticRecordKind::ACCEPTED_OUTPUT_TOKEN);
    assert_eq!(output_token.sequence_id, 0);
    assert_eq!(output_token.scheduled_tokens, 3838);
    assert_eq!(end.kind, SemanticRecordKind::ENGINE_STEP_END);
    assert!(reader.try_next().unwrap().is_none());
    assert_eq!(reader.dropped_records(), 0);

    drop(reader);
    unsafe {
        gpu_observer_bridge_close(bridge);
    }
    fs::remove_file(path).unwrap();
}

/// Exercises the live per-request focus mechanism end-to-end: EngineCore's
/// own bridge (created via gpu_observer_bridge_open, the writer of
/// everything else) seeds a static focus at startup, then a second,
/// independently-opened handle -- standing in for the vLLM frontend process,
/// which does NOT create the ring -- sets a new dynamic focus via
/// gpu_observer_bridge_focus_open/_set. EngineCore's original bridge handle
/// must observe the update live, with no re-open, no truncation, and without
/// disturbing the ring's head/tail/dropped cursors (the exact hazard the
/// non-destructive SemanticFocusHandle type exists to avoid -- see its
/// doc comment in vllm-adapter/native/src/lib.rs).
#[test]
fn frontend_process_sets_live_focus_without_disturbing_the_ring() {
    let path = std::env::temp_dir().join(format!(
        "gpu-observer-focus-test-{}-{}.ring",
        std::process::id(),
        std::thread::current().name().unwrap_or("unnamed")
    ));
    let path_string = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();

    // EngineCore creates the ring and seeds a static focus (backward
    // compatibility with GPU_OBSERVER_FOCUS_REQUEST_ID offline captures).
    let bridge = unsafe { gpu_observer_bridge_open(path_string.as_ptr(), 64) };
    assert!(!bridge.is_null());
    assert_eq!(unsafe { gpu_observer_bridge_set_focus(bridge, 0xAAAA) }, 0);
    assert_eq!(unsafe { gpu_observer_bridge_get_focus(bridge) }, 0xAAAA);

    // Publish one record so the ring has real head/tail/dropped state to
    // prove the focus-only handle never disturbs.
    let input = SemanticSliceInput {
        request_id: 0x1234,
        sequence_id: 0,
        scheduled_tokens: 10,
        service_class_id: 0,
        phase: RequestPhase::Prefill as u8,
        reserved: 0,
    };
    assert_eq!(
        unsafe { gpu_observer_emit_step_begin(bridge, 100, 1, 10, 10, 0, 0, 1, 0, &input, 1) },
        0
    );

    // A second process (the frontend) opens the SAME already-created file
    // through the non-destructive focus-only path and sets a new, dynamic
    // focus -- simulating a brand-new chat message arriving.
    let focus_handle = unsafe { gpu_observer_bridge_focus_open(path_string.as_ptr()) };
    assert!(!focus_handle.is_null());
    assert_eq!(unsafe { gpu_observer_bridge_focus_set(focus_handle, 0xBEEF) }, 0);
    unsafe {
        gpu_observer_bridge_focus_close(focus_handle);
    }

    // EngineCore's original, still-open bridge handle observes the new
    // value live, with no re-open.
    assert_eq!(unsafe { gpu_observer_bridge_get_focus(bridge) }, 0xBEEF);

    // The records published before the focus change are still there, intact,
    // at the same sequence numbers -- the focus-only handle did not truncate
    // or otherwise disturb the ring. emit_step_begin with slice_count=1
    // publishes the begin record plus its one bundled request_slice
    // atomically (see reserve()'s `required = slice_count + 1`), so both are
    // expected here.
    let mut reader = SemanticRingReader::open(&path).unwrap();
    let begin = reader.try_next().unwrap().unwrap();
    assert_eq!(begin.kind, SemanticRecordKind::ENGINE_STEP_BEGIN);
    assert_eq!(begin.sequence, 0);
    let slice = reader.try_next().unwrap().unwrap();
    assert_eq!(slice.kind, SemanticRecordKind::STEP_REQUEST_SLICE);
    assert_eq!(slice.sequence, 1);
    assert!(reader.try_next().unwrap().is_none());
    assert_eq!(reader.dropped_records(), 0);

    drop(reader);
    unsafe {
        gpu_observer_bridge_close(bridge);
    }
    fs::remove_file(path).unwrap();
}
