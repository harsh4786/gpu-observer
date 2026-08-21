//! Live WebSocket tail of the CUPTI activity agent's `activities.tsv`
//! (`cupti-agent/activity/cupti_activity_agent.cpp`) -- real per-kernel-launch
//! records (name, grid/block dims, stream, correlation id, actual GPU
//! start/end) for every kernel the process launches, not just one anchor
//! kernel. This is a genuinely different file-lifecycle than the sanitizer's:
//! `activities.tsv` is opened once (`"w"`) and grows append-only for the life
//! of the capture -- the agent never rewrites it -- so this tailer just
//! tracks a byte offset and parses newly-appended complete lines, buffering
//! any trailing partial line (the agent may be mid-`fprintf` when we read)
//! until the next poll completes it. No re-signal, no dedup-by-identity
//! needed, unlike `sanitizer_stream_server.rs`.
//!
//! This server does no kernel-name classification -- it forwards raw fields
//! as-is. Mapping a launch to one of Qwen3's 11 per-layer stages (name is
//! ambiguous for four of them: qkv_proj/o_proj/gate_up_proj/down_proj all
//! share the `gemvx::kernel` symbol during decode, disambiguated only by
//! grid shape and launch position) is presentation logic that belongs in
//! `ui/cupti-activity.js`, not here.
//!
//! Usage: cupti_stream_server ACTIVITIES_TSV_PATH BIND_ADDR

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::json;
use tungstenite::{Message, WebSocket};

const POLL_INTERVAL: Duration = Duration::from_millis(250);
const EXPECTED_FIELDS: usize = 18;

type Client = WebSocket<TcpStream>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();
    let program = arguments.next().unwrap_or_default();
    let usage = || {
        format!(
            "usage: {} ACTIVITIES_TSV_PATH BIND_ADDR",
            PathBuf::from(&program).display()
        )
    };
    let activities_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
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

    eprintln!("cupti_stream_server: tailing {}", activities_path.display());

    let mut offset: u64 = 0;
    let mut pending_partial = String::new();

    loop {
        match tail_new_lines(&activities_path, &mut offset, &mut pending_partial) {
            Ok(lines) => {
                for line in lines {
                    if let Some(patch) = parse_line(&line) {
                        broadcast(&clients, &patch.to_string());
                    }
                }
            }
            Err(_) => {
                // File not created yet (CUPTI not armed) or a transient read
                // race: skip this tick, retry next -- same fail-open stance
                // sanitizer_stream_server.rs takes for mid-flush races.
            }
        }
        thread::sleep(POLL_INTERVAL);
    }
}

/// Reads any bytes appended since `*offset`, returning complete new lines
/// (header/comment lines included -- filtered out in `parse_line`). Advances
/// `*offset` past everything consumed, including a completed pending partial
/// line from the previous call.
fn tail_new_lines(
    path: &PathBuf,
    offset: &mut u64,
    pending_partial: &mut String,
) -> std::io::Result<Vec<String>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len < *offset {
        // The file was recreated (fresh capture run): restart from scratch
        // rather than seeking backward into a different file's bytes.
        *offset = 0;
        pending_partial.clear();
    }
    if len == *offset {
        return Ok(Vec::new());
    }
    file.seek(SeekFrom::Start(*offset))?;
    let mut chunk = String::new();
    file.read_to_string(&mut chunk)?;
    *offset = len;

    pending_partial.push_str(&chunk);
    let mut lines = Vec::new();
    loop {
        match pending_partial.find('\n') {
            Some(index) => {
                let line = pending_partial[..index].to_string();
                *pending_partial = pending_partial[index + 1..].to_string();
                lines.push(line);
            }
            None => break,
        }
    }
    Ok(lines)
}

fn parse_line(line: &str) -> Option<serde_json::Value> {
    if line.starts_with('#') || line.starts_with("start_ns\t") {
        return None; // comment header or column-name header line
    }
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() != EXPECTED_FIELDS {
        return None; // truncated/malformed line; skip rather than crash the tailer
    }
    let start_ns = fields[0];
    let end_ns = fields[1];
    let device: u32 = fields[2].parse().ok()?;
    let context: u32 = fields[3].parse().ok()?;
    let stream: u32 = fields[4].parse().ok()?;
    let correlation: u32 = fields[5].parse().ok()?;
    let grid_id = fields[6]; // CUPTI's own per-launch id, not a grid dimension -- kept as string (i64)
    let graph_node_id = fields[7]; // u64, string
    let graph_id: u32 = fields[8].parse().ok()?;
    let grid_x: i64 = fields[9].parse().ok()?;
    let grid_y: i64 = fields[10].parse().ok()?;
    let grid_z: i64 = fields[11].parse().ok()?;
    let block_x: i64 = fields[12].parse().ok()?;
    let block_y: i64 = fields[13].parse().ok()?;
    let block_z: i64 = fields[14].parse().ok()?;
    let channel_id: u32 = fields[15].parse().ok()?;
    let channel_type: u32 = fields[16].parse().ok()?;
    let name = fields[17];

    Some(json!({
        "kind": "kernel_launch",
        "name": name,
        "startNs": start_ns,
        "endNs": end_ns,
        "device": device,
        "context": context,
        "stream": stream,
        "correlationId": correlation,
        "launchGridId": grid_id,
        "graphNodeId": graph_node_id,
        "graphId": graph_id,
        "grid": [grid_x, grid_y, grid_z],
        "block": [block_x, block_y, block_z],
        "channelId": channel_id,
        "channelType": channel_type,
    }))
}

fn spawn_accept_loop(
    bind_addr: String,
    clients: Arc<Mutex<Vec<Client>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(&bind_addr)?;
    eprintln!("cupti_stream_server: listening on ws://{bind_addr}");
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            match tungstenite::accept(stream) {
                Ok(socket) => {
                    if let Err(error) = socket.get_ref().set_nonblocking(true) {
                        eprintln!("cupti_stream_server: set_nonblocking failed: {error}");
                        continue;
                    }
                    clients.lock().unwrap().push(socket);
                }
                Err(error) => {
                    eprintln!("cupti_stream_server: WS handshake failed: {error}");
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
