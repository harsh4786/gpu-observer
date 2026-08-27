//! The only place this crate writes to the DOM. Every function here reads
//! the last-written value out of `AppState::render_cache` first and skips
//! the actual `web_sys` call when nothing changed -- this is the real
//! "fewer FFI crossings" win: not a novel memory format, just never
//! touching a DOM node with the value it already has.

use web_sys::{Document, Element};

use crate::state::{AppState, MAX_ROWS, ROLLING_WINDOW_NS};

fn doc() -> Document {
    web_sys::window().expect("no global window").document().expect("no document")
}

fn by_id(id: &str) -> Option<Element> {
    doc().get_element_by_id(id)
}

fn set_text_if_changed(id: &str, text: &str, cache: &mut Option<String>) {
    if cache.as_deref() == Some(text) {
        return;
    }
    if let Some(el) = by_id(id) {
        el.set_text_content(Some(text));
    }
    *cache = Some(text.to_string());
}

pub fn set_reply_text(text: &str) {
    if let Some(el) = by_id("chat-reply") {
        el.set_text_content(Some(text));
    }
}

pub fn set_waiting_text(text: &str) {
    if let Some(el) = by_id("chat-waiting") {
        el.set_text_content(Some(text));
    }
}

pub fn set_send_disabled(disabled: bool) {
    if let Some(el) = by_id("chat-send") {
        let _ = el.toggle_attribute_with_force("disabled", disabled);
    }
}

pub fn set_stop_disabled(disabled: bool) {
    if let Some(el) = by_id("chat-stop") {
        let _ = el.toggle_attribute_with_force("disabled", disabled);
    }
}

fn set_status(id: &str, connected: bool, cache: &mut Option<bool>) {
    if *cache == Some(connected) {
        return;
    }
    if let Some(el) = by_id(id) {
        el.set_text_content(Some(if connected { "connected" } else { "connecting…" }));
        let class_list = el.class_list();
        let _ = class_list.toggle_with_force("connected", connected);
        let _ = class_list.toggle_with_force("disconnected", !connected);
    }
    *cache = Some(connected);
}

/// The periodic (150ms) stats tick -- kernel bar list, now-executing line,
/// rolling GPU-busy bar, prompt-token count, connection pills. Called with
/// `&mut AppState` because computing the rolling-busy window prunes
/// `recent_launches` in place (matches `ui/cupti-activity.js`'s
/// `getRollingBusy`, which does the same filtering as a side effect of
/// reading it).
pub fn render_stats(state: &mut AppState) {
    set_status("semantic-status", state.semantic_connected, &mut state.render_cache.semantic_connected);
    set_status("cupti-status", state.cupti_connected, &mut state.render_cache.cupti_connected);

    let now_executing = state.now_executing.clone().unwrap_or_else(|| "waiting for the first real launch…".to_string());
    set_text_if_changed("now-executing", &now_executing, &mut state.render_cache.now_executing);

    if state.render_cache.prompt_tokens != state.prompt_tokens {
        let prompt_text = match state.prompt_tokens {
            Some(n) => format!("{n} tokens"),
            None => "— tokens".to_string(),
        };
        if let Some(el) = by_id("prompt-tokens") {
            el.set_text_content(Some(&prompt_text));
        }
        state.render_cache.prompt_tokens = state.prompt_tokens;
    }

    render_kernel_rows(state);
    render_busy_bar(state);
}

fn render_kernel_rows(state: &mut AppState) {
    let (rows, overflow) = state.top_rows(MAX_ROWS);
    let rows_owned: Vec<(String, u32, u32)> =
        rows.iter().map(|(name, stat)| (name.to_string(), stat.thinking, stat.response)).collect();

    if state.render_cache.rows == rows_owned && state.render_cache.overflow_count == overflow {
        return; // nothing about the visible list actually changed since last tick
    }

    if let Some(container) = by_id("kernel-rows") {
        container.set_inner_html("");
        let max_total = rows_owned.iter().map(|(_, t, r)| t + r).max().unwrap_or(1).max(1);
        let document = doc();
        for (name, thinking, response) in &rows_owned {
            let total = thinking + response;
            let row = document.create_element("div").unwrap();
            row.set_class_name("kernel-row");

            let label = document.create_element("span").unwrap();
            label.set_class_name("kernel-name");
            label.set_text_content(Some(name));
            row.append_child(&label).ok();

            let bar = document.create_element("div").unwrap();
            bar.set_class_name("kernel-bar");
            if *thinking > 0 {
                let seg = document.create_element("div").unwrap();
                seg.set_class_name("kernel-bar-thinking");
                let width = (*thinking as f64 / max_total as f64) * 100.0;
                seg.set_attribute("style", &format!("width:{width:.2}%")).ok();
                bar.append_child(&seg).ok();
            }
            if *response > 0 {
                let seg = document.create_element("div").unwrap();
                seg.set_class_name("kernel-bar-response");
                let width = (*response as f64 / max_total as f64) * 100.0;
                seg.set_attribute("style", &format!("width:{width:.2}%")).ok();
                bar.append_child(&seg).ok();
            }
            row.append_child(&bar).ok();

            let count = document.create_element("span").unwrap();
            count.set_class_name("kernel-count");
            count.set_text_content(Some(&total.to_string()));
            row.append_child(&count).ok();

            container.append_child(&row).ok();
        }
    }

    let overflow_text = if overflow > 0 { format!("+{overflow} more distinct kernels") } else { String::new() };
    if let Some(el) = by_id("kernel-overflow") {
        el.set_text_content(Some(&overflow_text));
    }

    state.render_cache.rows = rows_owned;
    state.render_cache.overflow_count = overflow;
}

fn render_busy_bar(state: &mut AppState) {
    let (busy_ns, window_ns, launch_count) = state.rolling_busy();
    if launch_count == 0 {
        if state.render_cache.busy_percent != Some(-1) {
            if let Some(el) = by_id("busy-caption") {
                el.set_text_content(Some("waiting for the first real launch…"));
            }
            if let Some(el) = by_id("busy-bar-fill") {
                el.set_attribute("style", "width:0%").ok();
            }
            state.render_cache.busy_percent = Some(-1);
        }
        return;
    }

    let busy_ms = busy_ns as f64 / 1e6;
    let window_ms = window_ns as f64 / 1e6;
    let fraction = if window_ms > 0.0 { (busy_ms / window_ms).min(1.0) } else { 0.0 };
    let percent = (fraction * 100.0).round() as i32;

    if state.render_cache.busy_percent != Some(percent) {
        if let Some(el) = by_id("busy-bar-fill") {
            el.set_attribute("style", &format!("width:{percent}%")).ok();
        }
        state.render_cache.busy_percent = Some(percent);
    }

    let window_s = ROLLING_WINDOW_NS as f64 / 1e9;
    let caption = format!(
        "{busy_ms:.1} ms GPU-busy of the last {window_s:.1}s of device activity ({percent}%) · {launch_count} launches"
    );
    if state.render_cache.busy_caption.as_deref() != Some(caption.as_str()) {
        if let Some(el) = by_id("busy-caption") {
            el.set_text_content(Some(&caption));
        }
        state.render_cache.busy_caption = Some(caption);
    }
}
