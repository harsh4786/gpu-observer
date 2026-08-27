//! Entry point for the Qwen3.8-27B live UI. Almost everything -- WebSocket
//! handling, JSON parsing, per-kernel-name aggregation, the thinking/
//! response phase state machine, rolling-GPU-busy math, and the actual DOM
//! writes -- lives here in Rust via `web-sys`, not a hand-written JS glue
//! layer. See `global.rs` for why the shared state is a raw `static`
//! instead of `Rc<RefCell<_>>`, and `render.rs` for the diff-before-write
//! DOM update discipline.

mod chat;
mod global;
mod render;
mod state;
mod ws;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Event, HtmlInputElement, KeyboardEvent};

// Distinct from ui/'s 8000/8088-8090 scheme so the two stacks never
// collide, even though this box's memory means they can't run
// concurrently anyway (see the plan).
const SEMANTIC_WS_PORT: u16 = 9190;
const CUPTI_WS_PORT: u16 = 9191;
const RENDER_INTERVAL_MS: i32 = state::RENDER_INTERVAL_MS;

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    global::init();

    ws::connect_semantic(SEMANTIC_WS_PORT);
    ws::connect_cupti(CUPTI_WS_PORT);

    wire_chat_form()?;
    wire_stop_button()?;
    wire_escape_key()?;
    start_render_loop()?;

    Ok(())
}

fn document() -> web_sys::Document {
    web_sys::window().expect("no global window").document().expect("no document")
}

fn wire_chat_form() -> Result<(), JsValue> {
    let form = document()
        .get_element_by_id("chat-form")
        .ok_or_else(|| JsValue::from_str("missing #chat-form"))?;
    let input = document()
        .get_element_by_id("chat-input")
        .ok_or_else(|| JsValue::from_str("missing #chat-input"))?
        .dyn_into::<HtmlInputElement>()?;

    let closure = Closure::<dyn FnMut(Event)>::new(move |event: Event| {
        event.prevent_default();
        let text = input.value();
        input.set_value("");
        chat::send_message(text);
    });
    form.add_event_listener_with_callback("submit", closure.as_ref().unchecked_ref())?;
    closure.forget();
    Ok(())
}

fn wire_stop_button() -> Result<(), JsValue> {
    let button = document()
        .get_element_by_id("chat-stop")
        .ok_or_else(|| JsValue::from_str("missing #chat-stop"))?;
    let closure = Closure::<dyn FnMut()>::new(chat::stop);
    button.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
    closure.forget();
    Ok(())
}

/// Escape as the cancel shortcut, matching `ui/trace.js`: browsers reserve
/// Ctrl+C for copy, so Escape is the standard "cancel the in-flight thing"
/// pattern instead.
fn wire_escape_key() -> Result<(), JsValue> {
    let closure = Closure::<dyn FnMut(KeyboardEvent)>::new(move |event: KeyboardEvent| {
        if event.key() == "Escape" {
            chat::stop();
        }
    });
    document().add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())?;
    closure.forget();
    Ok(())
}

fn start_render_loop() -> Result<(), JsValue> {
    let closure = Closure::<dyn FnMut()>::new(|| {
        render::render_stats(global::state());
    });
    web_sys::window()
        .expect("no global window")
        .set_interval_with_callback_and_timeout_and_arguments_0(
            closure.as_ref().unchecked_ref(),
            RENDER_INTERVAL_MS,
        )?;
    closure.forget();
    Ok(())
}
