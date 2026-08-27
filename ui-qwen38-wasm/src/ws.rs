//! The two live WebSocket connections: the semantic ring (real-time engine-
//! step/request scheduling patches -- we only need `request_slice`'s real
//! `tokens` field for the prompt-token-count feature) and the CUPTI stream
//! (real per-launch `kernel_launch` events -- the generic feed's only data
//! source, no per-architecture classifier).
//!
//! Each `onmessage` closure hands the raw message string straight to
//! `serde_json::from_str`, one parse, directly into the global `AppState`
//! -- no intermediate JS object, no second parse.
//!
//! Every closure fetches `global::state()` exactly once and passes the
//! resulting `&mut AppState` straight into a plain function -- never twice
//! within the same call chain, since two live `&'static mut` handles to
//! the same cell would be real aliasing UB even in this single-threaded
//! model (see `global.rs`'s doc comment).

use serde_json::Value;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, WebSocket};

use crate::global;
use crate::state::AppState;

const RECONNECT_DELAY_MS: i32 = 2000;

fn ws_url(port: u16) -> String {
    let window = web_sys::window().expect("no global window");
    let hostname = window.location().hostname().unwrap_or_else(|_| "127.0.0.1".into());
    format!("ws://{hostname}:{port}")
}

fn set_timeout(callback: &Closure<dyn FnMut()>, ms: i32) {
    if let Some(window) = web_sys::window() {
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.as_ref().unchecked_ref(),
            ms,
        );
    }
}

/// Semantic ring: only `request_slice` is consumed (for the real prompt
/// token count); everything else is ignored, matching this UI's
/// deliberately minimal scope (no scheduler-slice strip, no packed-row
/// diagram -- those are visual features this stack doesn't carry over).
pub fn connect_semantic(port: u16) {
    connect(port, true);
}

/// CUPTI stream: `kernel_launch` events feed the generic per-name tally.
pub fn connect_cupti(port: u16) {
    connect(port, false);
}

fn connect(port: u16, is_semantic: bool) {
    let ws = match WebSocket::new(&ws_url(port)) {
        Ok(ws) => ws,
        Err(_) => {
            schedule_reconnect(port, is_semantic);
            return;
        }
    };

    let onopen = Closure::<dyn FnMut()>::new(move || {
        let state = global::state();
        if is_semantic {
            state.semantic_connected = true;
        } else {
            state.cupti_connected = true;
        }
    });
    ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
    onopen.forget();

    let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
        if let Some(text) = event.data().as_string() {
            handle_message(global::state(), is_semantic, &text);
        }
    });
    ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();

    let onclose = Closure::<dyn FnMut()>::new(move || {
        let state = global::state();
        if is_semantic {
            state.semantic_connected = false;
        } else {
            state.cupti_connected = false;
        }
        schedule_reconnect(port, is_semantic);
    });
    ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
    onclose.forget();

    let ws_for_error = ws.clone();
    let onerror = Closure::<dyn FnMut()>::new(move || {
        let _ = ws_for_error.close();
    });
    ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
    onerror.forget();
}

fn schedule_reconnect(port: u16, is_semantic: bool) {
    let callback = Closure::<dyn FnMut()>::new(move || {
        connect(port, is_semantic);
    });
    set_timeout(&callback, RECONNECT_DELAY_MS);
    callback.forget();
}

fn handle_message(state: &mut AppState, is_semantic: bool, text: &str) {
    let value: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return, // a malformed frame must never take down the live feed
    };
    let kind = value.get("kind").and_then(Value::as_str).unwrap_or("");

    if is_semantic {
        if kind == "request_slice" && state.prompt_tokens.is_none() {
            if let Some(tokens) = value.get("tokens").and_then(Value::as_u64) {
                state.prompt_tokens = Some(tokens as u32);
            }
        }
        return;
    }

    if kind != "kernel_launch" {
        return;
    }
    let Some(name) = value.get("name").and_then(Value::as_str) else { return };
    let start_ns = value.get("startNs").and_then(Value::as_i64).unwrap_or(0);
    let end_ns = value.get("endNs").and_then(Value::as_i64).unwrap_or(0);
    state.record_launch(name, start_ns, end_ns);
}
