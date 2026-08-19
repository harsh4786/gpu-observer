//! Live WebSocket tail of the vLLM semantic ring.
//!
//! This is `semantic_capture.rs`'s poll loop with two structural changes: the
//! per-record output goes to connected WebSocket clients instead of a flat
//! file, and there is no fixed capture duration -- it runs for the life of
//! the process. Everything else (SemanticRingReader::try_next, the 200us
//! idle poll, drop accounting) is unchanged from the existing offline tool.
//!
//! Records are re-serialized to small JSON objects with semantically named
//! fields (not raw wire-field names -- e.g. PACKED_REQUEST_SLICE's
//! prefill_tokens/decode_tokens really mean row_begin/row_end, see
//! observer-core/src/semantic.rs's constructors) so the browser never has to
//! know the wire layout. u64 values that can exceed JavaScript's 2^53 safe-
//! integer range (timestamps, sequence numbers, request hashes, step IDs)
//! are sent as decimal strings, not JSON numbers.

use std::env;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use gpu_observer_collector::semantic::SemanticRingReader;
use gpu_observer_core::{SemanticRecordKind, SemanticWireRecord};
use serde_json::{json, Value};
use tungstenite::{Message, WebSocket};

const IDLE_POLL_MICROS: u64 = 200;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);

type Client = WebSocket<TcpStream>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let usage = || {
        format!(
            "usage: {} SHARED_RING BIND_ADDR",
            PathBuf::from(&program).display()
        )
    };
    let shared_ring = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let bind_addr = arguments
        .next()
        .ok_or_else(&usage)?
        .to_string_lossy()
        .into_owned();
    if arguments.next().is_some() {
        return Err(usage().into());
    }

    let clients: Arc<Mutex<Vec<Client>>> = Arc::new(Mutex::new(Vec::new()));
    spawn_accept_loop(bind_addr, Arc::clone(&clients))?;

    let mut reader = SemanticRingReader::open(&shared_ring)?;
    eprintln!("semantic_stream_server: tailing {}", shared_ring.display());
    let mut last_heartbeat = Instant::now();

    loop {
        let mut drained = false;
        while let Some(record) = reader.try_next()? {
            drained = true;
            let patch = record_to_patch(record);
            broadcast(&clients, &patch.to_string());
        }
        if !drained {
            thread::sleep(Duration::from_micros(IDLE_POLL_MICROS));
        }
        // Cheap periodic liveness signal so a freshly connected client can
        // tell the transport is alive even during genuine idle gaps between
        // engine steps, without waiting on the next real record. Timer-based,
        // not a modulo on a counter that stops advancing while idle -- that
        // earlier version re-fired on every single idle poll tick once
        // `records` landed on a multiple of the modulus (including 0).
        if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
            last_heartbeat = Instant::now();
            broadcast(
                &clients,
                &json!({"kind": "dropped_total", "count": reader.dropped_records().to_string()})
                    .to_string(),
            );
        }
    }
}

fn spawn_accept_loop(
    bind_addr: String,
    clients: Arc<Mutex<Vec<Client>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(&bind_addr)?;
    eprintln!("semantic_stream_server: listening on ws://{bind_addr}");
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            // Handshake blocking (fast, one round trip); only the
            // post-handshake broadcast path needs non-blocking writes so one
            // slow client can never stall the ring drain loop.
            match tungstenite::accept(stream) {
                Ok(socket) => {
                    if let Err(error) = socket.get_ref().set_nonblocking(true) {
                        eprintln!("semantic_stream_server: set_nonblocking failed: {error}");
                        continue;
                    }
                    clients.lock().unwrap().push(socket);
                }
                Err(error) => {
                    eprintln!("semantic_stream_server: WS handshake failed: {error}");
                }
            }
        }
    });
    Ok(())
}

fn broadcast(clients: &Arc<Mutex<Vec<Client>>>, text: &str) {
    let mut guard = clients.lock().unwrap();
    if guard.is_empty() {
        return;
    }
    guard.retain_mut(|client| match client.write(Message::text(text)) {
        Ok(()) => matches!(client.flush(), Ok(()) | Err(tungstenite::Error::Io(_))),
        Err(tungstenite::Error::Io(error)) if error.kind() == std::io::ErrorKind::WouldBlock => {
            true // best-effort live stream: skip this message for a slow client, keep it connected
        }
        Err(_) => false, // client actually disconnected
    });
}

fn record_to_patch(record: SemanticWireRecord) -> Value {
    let ts = record.timestamp_ns.to_string();
    let seq = record.sequence.to_string();
    let step = record.step_id.to_string();
    match record.kind {
        SemanticRecordKind::ENGINE_STEP_BEGIN => json!({
            "kind": "step_begin",
            "ts": ts, "seq": seq, "step": step,
            "scheduled": record.scheduled_tokens,
            "prefill": record.prefill_tokens,
            "decode": record.decode_tokens,
            "queue": record.queue_depth,
            "active": record.active_requests,
            "slices": record.expected_slices,
            "kv_permyriad": record.kv_cache_usage_permyriad,
            "flags": record.flags,
        }),
        SemanticRecordKind::STEP_REQUEST_SLICE => json!({
            "kind": "request_slice",
            "ts": ts, "seq": seq, "step": step,
            "request": format!("{:016x}", record.request_id),
            "phase": record.phase,
            "tokens": record.scheduled_tokens,
        }),
        SemanticRecordKind::PACKED_LAYOUT_BEGIN => json!({
            "kind": "packed_layout_begin",
            "ts": ts, "seq": seq, "step": step,
            "generation": record.sequence_id.to_string(),
            "tokens": record.scheduled_tokens,
            "slices": record.expected_slices,
            "flags": record.flags,
        }),
        SemanticRecordKind::PACKED_REQUEST_SLICE => json!({
            "kind": "packed_request_slice",
            "ts": ts, "seq": seq, "step": step,
            "generation": record.sequence_id.to_string(),
            "request": format!("{:016x}", record.request_id),
            "packed_index": record.queue_depth,
            "row_begin": record.prefill_tokens,
            "row_end": record.decode_tokens,
            "tokens": record.scheduled_tokens,
            "phase": record.phase,
        }),
        SemanticRecordKind::PACKED_TOKEN_ROW => json!({
            "kind": "packed_token_row",
            "ts": ts, "seq": seq, "step": step,
            "generation": record.sequence_id.to_string(),
            "request": format!("{:016x}", record.request_id),
            "packed_row": record.prefill_tokens,
            "sequence_position": record.decode_tokens,
            "token_id": record.scheduled_tokens,
            "phase": record.phase,
            "flags": record.flags,
        }),
        SemanticRecordKind::ACCEPTED_OUTPUT_TOKEN => json!({
            "kind": "accepted_output_token",
            "ts": ts, "seq": seq, "step": step,
            "request": format!("{:016x}", record.request_id),
            "output_position": record.sequence_id,
            "token_id": record.scheduled_tokens,
            "flags": record.flags,
        }),
        SemanticRecordKind::ENGINE_STEP_END => json!({
            "kind": "step_end",
            "ts": ts, "seq": seq, "step": step,
            "status": record.status,
            "flags": record.flags,
        }),
        other => json!({"kind": "unknown", "raw_kind": other, "ts": ts, "seq": seq, "step": step}),
    }
}
