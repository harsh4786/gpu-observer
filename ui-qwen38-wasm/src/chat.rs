//! Chat send/stream/abort. Mirrors `ui/trace.js`'s `sendChatMessage`: POST
//! streaming chat completion, read the SSE body incrementally, and watch
//! the accumulated reply text for Qwen's own `<think>`/`</think>` tags to
//! drive the thinking/response split -- same reconstructed-correlation
//! caveat as the original (`ui/cupti-activity.js`'s `inThinkingPhase`
//! comment): the boundary comes from this text stream, kernel launches come
//! from a separate WebSocket, so a launch's bucket is "whichever phase was
//! current when it was processed," not a measured per-token attribution.

use std::cell::UnsafeCell;

use js_sys::{Reflect, Uint8Array};
use serde_json::json;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{AbortController, Request, RequestInit, RequestMode, Response, TextDecoder};

use crate::global;
use crate::render;

const MODEL_NAME: &str = "Qwen/Qwen3.8-27B";

// The in-flight request's AbortController, if any -- same single-threaded
// justification as global.rs's AppState cell. Kept separate from AppState
// (rather than adding a web_sys type to it) so state.rs stays plain data
// with no web-sys dependency.
struct AbortCell(UnsafeCell<Option<AbortController>>);
// SAFETY: see global.rs's module doc -- single-threaded execution model.
unsafe impl Sync for AbortCell {}
static ABORT: AbortCell = AbortCell(UnsafeCell::new(None));

fn set_abort(controller: Option<AbortController>) {
    // SAFETY: single-threaded; no other live reference to this cell exists
    // across this assignment.
    unsafe {
        *ABORT.0.get() = controller;
    }
}

fn take_abort() -> Option<AbortController> {
    // SAFETY: single-threaded; the returned value is a cheap Clone of a JS
    // handle (an owned reference-counted binding), not a borrow of the
    // cell's storage.
    unsafe { (*ABORT.0.get()).clone() }
}

fn vllm_base() -> String {
    let window = web_sys::window().expect("no global window");
    let hostname = window.location().hostname().unwrap_or_else(|_| "127.0.0.1".into());
    format!("http://{hostname}:9000")
}

pub fn send_message(text: String) {
    let text = text.trim().to_string();
    let state = global::state();
    if text.is_empty() || state.chat_busy {
        return;
    }
    state.chat_busy = true;
    state.reset_query();
    render::set_waiting_text("Request sent — waiting for the scheduler to admit it…");
    render::set_send_disabled(true);
    render::set_stop_disabled(false);
    render::set_reply_text("");

    let controller = AbortController::new().ok();
    set_abort(controller.clone());

    wasm_bindgen_futures::spawn_local(async move {
        let result = run_stream(text, controller.as_ref()).await;
        if let Err(err) = result {
            let message = err.as_string().unwrap_or_default();
            let aborted = message.contains("Abort") || message.contains("abort");
            render::set_waiting_text(if aborted {
                "Stopped — generation cancelled server-side."
            } else {
                "Live request failed."
            });
        }
        global::state().chat_busy = false;
        set_abort(None);
        render::set_send_disabled(false);
        render::set_stop_disabled(true);
    });
}

pub fn stop() {
    if let Some(controller) = take_abort() {
        controller.abort();
    }
}

async fn run_stream(text: String, controller: Option<&AbortController>) -> Result<(), JsValue> {
    let body = json!({
        "model": MODEL_NAME,
        "messages": [{ "role": "user", "content": text }],
        "stream": true,
    })
    .to_string();

    let opts = RequestInit::new();
    opts.set_method("POST");
    opts.set_mode(RequestMode::Cors);
    opts.set_body(&JsValue::from_str(&body));
    if let Some(controller) = controller {
        opts.set_signal(Some(&controller.signal()));
    }

    let url = format!("{}/v1/chat/completions", vllm_base());
    let request = Request::new_with_str_and_init(&url, &opts)?;
    request.headers().set("Content-Type", "application/json")?;

    let window = web_sys::window().expect("no global window");
    let response: Response = JsFuture::from(window.fetch_with_request(&request))
        .await?
        .dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str(&format!("HTTP {}", response.status())));
    }
    let body = response.body().ok_or_else(|| JsValue::from_str("no response body"))?;
    let reader = body.get_reader().dyn_into::<web_sys::ReadableStreamDefaultReader>()?;
    let decoder = TextDecoder::new()?;

    let mut buffer = String::new();
    loop {
        let chunk = JsFuture::from(reader.read()).await?;
        let done = Reflect::get(&chunk, &JsValue::from_str("done"))?
            .as_bool()
            .unwrap_or(true);
        if done {
            break;
        }
        let value = Reflect::get(&chunk, &JsValue::from_str("value"))?;
        let bytes: Uint8Array = value.dyn_into()?;
        let text_chunk = decoder.decode_with_buffer_source(&bytes)?;
        buffer.push_str(&text_chunk);

        while let Some(pos) = buffer.find("\n\n") {
            let event: String = buffer.drain(..pos + 2).collect();
            handle_sse_event(&event);
        }
    }
    render::set_waiting_text("Response complete.");
    Ok(())
}

fn handle_sse_event(event: &str) {
    for line in event.lines() {
        let line = line.trim();
        let Some(payload) = line.strip_prefix("data:") else { continue };
        let payload = payload.trim();
        if payload == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(payload) else { continue };
        let delta = chunk
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str());
        if let Some(delta) = delta {
            let state = global::state();
            state.reply_text.push_str(delta);
            let has_open = state.reply_text.contains("<think>");
            let has_close = state.reply_text.contains("</think>");
            state.set_thinking(has_open && !has_close);
            render::set_reply_text(&state.reply_text);
        }
    }
}
