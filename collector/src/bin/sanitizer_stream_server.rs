//! Live WebSocket tail of the Compute Sanitizer device probe's
//! `reshape_and_cache_flash_kernel` block-entry events -- genuinely measured
//! real kernel launches, not the illustrative kernel-graph sweep the UI
//! otherwise shows. Only this one anchor kernel is patched; see the plan
//! this was built from for why (EXP-0009 through EXP-0018 proved exactly
//! this kernel safe under CUDA Graph replay with a specific env-var
//! configuration -- other kernels are not yet validated and are out of
//! scope here).
//!
//! Unlike semantic_stream_server.rs, this cannot simply tail a
//! monotonically-growing ring: the sanitizer's `flush_results()` in
//! subscriber.cpp OVERWRITES `*.events.bin` on every flush (opens with
//! `"wb"`), and the flush itself is lazy -- SIGUSR2 only sets a flag; the
//! actual write happens on the probe's next CUDA kernel launch. So this
//! tool periodically signals the EngineCore process, waits briefly for that
//! next-launch flush to land, re-reads the WHOLE file each time (reusing
//! `join_b_trace::load_device_events` unmodified -- the exact same parser
//! the project's own offline analysis tools use), and deduplicates against
//! events already broadcast by (launch_id, sequence). `sequence` alone is
//! only unique within one flush's retained set (it restarts at 0 per
//! header/context block -- see load_device_events), but `launch_id`
//! increments monotonically for the life of the EngineCore process, so the
//! pair is a safe process-lifetime dedup key: an event still retained
//! across two consecutive flushes has the same (launch_id, sequence) and is
//! correctly skipped; a genuinely new event has a new launch_id.
//!
//! Usage: sanitizer_stream_server CONTAINER_NAME EVENTS_BIN_PATH BIND_ADDR

use std::collections::HashSet;
use std::env;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use gpu_observer_collector::join_b_trace::load_device_events;
use serde_json::json;
use tungstenite::{Message, WebSocket};

const POLL_INTERVAL: Duration = Duration::from_millis(300);
const FLUSH_GRACE: Duration = Duration::from_millis(80);
const TARGET_KERNEL: &str = "reshape_and_cache_flash_kernel";

type Client = WebSocket<TcpStream>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let usage = || {
        format!(
            "usage: {} CONTAINER_NAME EVENTS_BIN_PATH BIND_ADDR",
            PathBuf::from(&program).display()
        )
    };
    let container = arguments
        .next()
        .ok_or_else(&usage)?
        .to_string_lossy()
        .into_owned();
    let events_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
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

    eprintln!(
        "sanitizer_stream_server: signaling EngineCore in container {container}, tailing {}",
        events_path.display()
    );

    let mut seen: HashSet<(u64, u32)> = HashSet::new();
    let mut total_forwarded: u64 = 0_u64;

    loop {
        if let Some(pid) = resolve_engine_pid(&container) {
            // Command::status() inherits the parent's stdio by default; this
            // process's own stdout/stderr are typically redirected to a log
            // file by the caller, so without explicit Stdio::null() here
            // that file silently fills with docker exec's own output
            // (observed in practice, not just theoretical) despite `kill`
            // having nothing useful to say on success.
            let _ = Command::new("docker")
                .args(["exec", &container, "kill", "-USR2", &pid])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        thread::sleep(FLUSH_GRACE);

        match load_device_events(&events_path) {
            Ok((events, quality)) => {
                for event in &events {
                    if seen.insert((event.launch_id, event.sequence)) {
                        total_forwarded += 1;
                        broadcast(&clients, &event_json(event).to_string());
                    }
                }
                if quality.drops > 0 || quality.sequence_errors > 0 {
                    broadcast(
                        &clients,
                        &json!({
                            "kind": "sanitizer_quality",
                            "drops": quality.drops.to_string(),
                            "sequence_errors": quality.sequence_errors.to_string(),
                            "total_forwarded": total_forwarded.to_string(),
                        })
                        .to_string(),
                    );
                }
            }
            Err(_) => {
                // Mid-flush race (file being rewritten right now) or the
                // probe hasn't flushed yet: skip this tick silently, retry
                // next. A read race is never a hard failure here -- the
                // next successful flush is a complete, consistent snapshot.
            }
        }

        thread::sleep(POLL_INTERVAL.saturating_sub(FLUSH_GRACE));
    }
}

fn resolve_engine_pid(container: &str) -> Option<String> {
    // Resolved fresh every poll rather than cached once: matches the exact
    // pattern every existing sanitizer run script uses
    // (benchmarks/run-sanitizer-cache-graph-arm.sh), and the PID must be
    // resolved from INSIDE the container's PID namespace (via `docker exec
    // ps`, not `docker top`) since `docker exec ... kill` targets that same
    // namespace.
    let output = Command::new("docker")
        .args([
            "exec",
            container,
            "sh",
            "-c",
            "ps -eo pid,comm | awk '$2 ~ /^VLLM::EngineCor/ {print $1; exit}'",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let pid = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if pid.is_empty() {
        None
    } else {
        Some(pid)
    }
}

fn event_json(event: &gpu_observer_collector::join_b_trace::DeviceEvent) -> serde_json::Value {
    json!({
        "kind": "kernel_block_event",
        "kernel": TARGET_KERNEL,
        "ts": event.device_timestamp_raw.to_string(),
        "launch_id": event.launch_id.to_string(),
        "sequence": event.sequence,
        "sm_id": event.sm_id,
        "block": [event.block[0], event.block[1], event.block[2]],
    })
}

fn spawn_accept_loop(
    bind_addr: String,
    clients: Arc<Mutex<Vec<Client>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(&bind_addr)?;
    eprintln!("sanitizer_stream_server: listening on ws://{bind_addr}");
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            match tungstenite::accept(stream) {
                Ok(socket) => {
                    if let Err(error) = socket.get_ref().set_nonblocking(true) {
                        eprintln!("sanitizer_stream_server: set_nonblocking failed: {error}");
                        continue;
                    }
                    clients.lock().unwrap().push(socket);
                }
                Err(error) => {
                    eprintln!("sanitizer_stream_server: WS handshake failed: {error}");
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
