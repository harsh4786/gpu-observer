//! The single global `AppState`, accessed via a raw `UnsafeCell` instead of
//! `Rc<RefCell<AppState>>`. This is a deliberate, scoped use of `unsafe`:
//! wasm running in a browser tab is single-threaded (there is exactly one
//! JS/wasm execution context; event-loop callbacks -- WS messages, fetch
//! resolutions, DOM events, the render interval -- always run one at a
//! time, never concurrently, and never re-enter each other mid-call), so
//! the aliasing check `RefCell` performs at runtime to catch concurrent
//! borrows can never actually be violated here. Bypassing it with
//! `UnsafeCell` removes real per-access overhead (a branch + a counter
//! read/write on every single field touch) for zero loss of the safety
//! `RefCell` would have provided in this specific execution model.
//!
//! This also removes the `Rc::clone()` boilerplate every closure needed
//! before: callbacks now just call `global::state()` fresh each time
//! instead of capturing a cloned handle.

use std::cell::UnsafeCell;

use crate::state::AppState;

struct StateCell(UnsafeCell<Option<AppState>>);

// SAFETY: see module doc -- single-threaded execution model means no two
// accesses to this cell's contents can ever be concurrent, so asserting
// Sync (required to make this a `static`) introduces no actual data race.
unsafe impl Sync for StateCell {}

static STATE: StateCell = StateCell(UnsafeCell::new(None));

/// Must be called exactly once, synchronously, at the very top of
/// `#[wasm_bindgen(start)]` -- before any closure is registered with the
/// browser (WebSocket/fetch/DOM listeners), so `state()` can never
/// possibly be called before this has run.
pub fn init() {
    // SAFETY: called once, synchronously, before the event loop can invoke
    // any other code path that might touch STATE -- no concurrent or
    // reentrant access is possible at this point.
    unsafe {
        *STATE.0.get() = Some(AppState::new());
    }
}

/// # Panics
/// If called before `init()` -- which cannot happen in practice (see
/// `init`'s contract), but is checked explicitly rather than assumed via
/// `unwrap_unchecked`, since this accessor is called from many call sites
/// and a clear panic message is worth the one branch.
pub fn state() -> &'static mut AppState {
    // SAFETY: single-threaded execution model (module doc) means this is
    // the only live reference to the contents at any given instant --
    // every call site finishes using the returned reference (a plain
    // function call, no held-across-await usage) before the next callback
    // can possibly run.
    unsafe { (*STATE.0.get()).as_mut().expect("global state accessed before init()") }
}
