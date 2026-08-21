//! Bounded localhost HTTP/SSE observer for EXP-0022.
//!
//! `std` is required for TCP and filesystem I/O. The measurement loop performs
//! no per-token allocation: network/SSE buffers reserve fixed caps at startup,
//! token IDs use a stack array, and fixed 96-byte records are written only after
//! the response completes.

use gpu_observer_core::{SemanticRecordKind, SemanticWireRecord, SEMANTIC_RECORD_BYTES};
use std::{
    env,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::PathBuf,
    time::Duration,
};

const MAX_BODY_BYTES: u64 = 1024 * 1024;
const MAX_HEADER_LINE_BYTES: usize = 16 * 1024;
const MAX_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_SSE_BUFFER_BYTES: usize = 2 * 1024 * 1024;
const MAX_EVENTS: usize = 8192;
const MAX_TOKENS_PER_FRAME: usize = 256;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = PathBuf::from(args.next().unwrap_or_default());
    let usage = || {
        format!(
            "usage: {} HOST PORT EXTERNAL_REQUEST_ID REQUEST.json OUTPUT.bin",
            program.display()
        )
    };
    let host = args
        .next()
        .ok_or_else(&usage)?
        .to_string_lossy()
        .into_owned();
    let port: u16 = args.next().ok_or_else(&usage)?.to_string_lossy().parse()?;
    let external_id = args
        .next()
        .ok_or_else(&usage)?
        .to_string_lossy()
        .into_owned();
    let body_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let output_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    if args.next().is_some() {
        return Err(usage().into());
    }

    let body_len = fs::metadata(&body_path)?.len();
    if body_len == 0 || body_len > MAX_BODY_BYTES {
        return Err(format!("request body size {body_len} is outside the bounded range").into());
    }
    let body = fs::read(&body_path)?;
    if !body.windows(13).any(|window| window == b"\"stream\":true")
        && !body.windows(14).any(|window| window == b"\"stream\": true")
    {
        return Err("request must set stream=true".into());
    }

    let frontend_id = format!("chatcmpl-{external_id}");
    let request_hash = stable_request_id(frontend_id.as_bytes());
    let pid = unsafe { libc::getpid() as u32 };
    let tid = unsafe { libc::syscall(libc::SYS_gettid) as u32 };
    let mut records = [SemanticWireRecord::default(); MAX_EVENTS];
    let mut record_count = 0_usize;

    let address = (host.as_str(), port)
        .to_socket_addrs()?
        .next()
        .ok_or("host did not resolve")?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(10))?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(900)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    let header = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nX-Request-Id: {external_id}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    let send_begin_ns = monotonic_ns()?;
    push_record(
        &mut records,
        &mut record_count,
        SemanticWireRecord::lifecycle(
            send_begin_ns,
            0,
            request_hash,
            0,
            0,
            0,
            0,
            SemanticRecordKind::CLIENT_REQUEST_SENT,
            0,
            pid,
            tid,
            0,
        ),
    )?;
    stream.write_all(header.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;

    let mut reader = BufReader::with_capacity(64 * 1024, stream);
    let mut line = String::with_capacity(1024);
    read_bounded_line(&mut reader, &mut line)?;
    let status = line
        .split_ascii_whitespace()
        .nth(1)
        .ok_or("HTTP response omitted status")?;
    if status != "200" {
        return Err(format!("HTTP status {status}").into());
    }
    let mut chunked = false;
    loop {
        read_bounded_line(&mut reader, &mut line)?;
        if line == "\r\n" || line == "\n" {
            break;
        }
        if line.to_ascii_lowercase().starts_with("transfer-encoding:")
            && line.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
    }
    if !chunked {
        return Err("expected a chunked SSE response".into());
    }

    let mut chunk = Vec::new();
    chunk.try_reserve_exact(MAX_CHUNK_BYTES)?;
    let mut sse = Vec::new();
    sse.try_reserve_exact(MAX_SSE_BUFFER_BYTES)?;
    let mut token_position = 0_u32;
    let mut saw_done = false;
    let mut completion_timestamp_ns = 0_u64;

    loop {
        read_bounded_line(&mut reader, &mut line)?;
        let size_text = line
            .trim()
            .split(';')
            .next()
            .ok_or("empty HTTP chunk size")?;
        let size = usize::from_str_radix(size_text, 16)?;
        if size == 0 {
            break;
        }
        if size > MAX_CHUNK_BYTES {
            return Err(format!("HTTP chunk exceeds {MAX_CHUNK_BYTES} bytes").into());
        }
        chunk.resize(size, 0);
        reader.read_exact(&mut chunk)?;
        let receipt_ns = monotonic_ns()?;
        let mut crlf = [0_u8; 2];
        reader.read_exact(&mut crlf)?;
        if crlf != *b"\r\n" {
            return Err("HTTP chunk missing CRLF terminator".into());
        }
        if sse.len().saturating_add(size) > MAX_SSE_BUFFER_BYTES {
            return Err("SSE framing buffer exceeded its bound".into());
        }
        sse.extend_from_slice(&chunk);

        let mut consumed = 0_usize;
        while let Some(relative_end) = find_bytes(&sse[consumed..], b"\n\n") {
            let frame_end = consumed + relative_end;
            let frame = &sse[consumed..frame_end];
            consumed = frame_end + 2;
            let data = frame
                .strip_prefix(b"data: ")
                .or_else(|| frame.strip_prefix(b"data:"))
                .unwrap_or(frame);
            if data == b"[DONE]" {
                saw_done = true;
                completion_timestamp_ns = receipt_ns;
                continue;
            }
            let mut token_ids = [0_u32; MAX_TOKENS_PER_FRAME];
            let token_count = parse_token_ids(data, &mut token_ids)?;
            for token_id in &token_ids[..token_count] {
                push_record(
                    &mut records,
                    &mut record_count,
                    SemanticWireRecord::lifecycle(
                        receipt_ns,
                        0,
                        request_hash,
                        0,
                        token_position,
                        *token_id,
                        0,
                        SemanticRecordKind::CLIENT_TOKEN_RECEIVED,
                        0,
                        pid,
                        tid,
                        0,
                    ),
                )?;
                token_position = token_position
                    .checked_add(1)
                    .ok_or("token position overflow")?;
            }
        }
        if consumed != 0 {
            sse.copy_within(consumed.., 0);
            sse.truncate(sse.len() - consumed);
        }
    }
    if !saw_done || !sse.is_empty() || token_position == 0 {
        return Err(format!(
            "incomplete SSE response: done={saw_done} buffered={} tokens={token_position}",
            sse.len()
        )
        .into());
    }
    push_record(
        &mut records,
        &mut record_count,
        SemanticWireRecord::lifecycle(
            completion_timestamp_ns,
            0,
            request_hash,
            0,
            token_position,
            0,
            0,
            SemanticRecordKind::CLIENT_REQUEST_COMPLETED,
            0,
            pid,
            tid,
            0,
        ),
    )?;

    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output_path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    for record in &records[..record_count] {
        writer.write_all(record.as_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    println!(
        "status=PASS request_hash=0x{request_hash:016x} tokens={} records={} send_begin_ns={} first_token_ns={} completed_ns={} output={}",
        token_position,
        record_count,
        send_begin_ns,
        records[1].timestamp_ns,
        completion_timestamp_ns,
        output_path.display(),
    );
    Ok(())
}

fn push_record(
    records: &mut [SemanticWireRecord; MAX_EVENTS],
    count: &mut usize,
    mut record: SemanticWireRecord,
) -> Result<(), Box<dyn Error>> {
    if *count >= records.len() {
        return Err("client event capacity exhausted".into());
    }
    record.sequence = *count as u64;
    records[*count] = record;
    *count += 1;
    Ok(())
}

fn read_bounded_line<R: BufRead>(reader: &mut R, line: &mut String) -> Result<(), Box<dyn Error>> {
    line.clear();
    let bytes = reader.read_line(line)?;
    if bytes == 0 {
        return Err("unexpected HTTP EOF".into());
    }
    if bytes > MAX_HEADER_LINE_BYTES {
        return Err("HTTP line exceeds bound".into());
    }
    Ok(())
}

fn parse_token_ids(data: &[u8], output: &mut [u32]) -> Result<usize, Box<dyn Error>> {
    let Some(start) = find_bytes(data, b"\"token_ids\":[") else {
        return Ok(0);
    };
    let mut cursor = start + b"\"token_ids\":[".len();
    let mut count = 0_usize;
    loop {
        while data.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if data.get(cursor) == Some(&b']') {
            return Ok(count);
        }
        if count == output.len() {
            return Err("SSE token list exceeds per-frame bound".into());
        }
        let begin = cursor;
        while data.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        if cursor == begin {
            return Err("invalid token ID in SSE frame".into());
        }
        output[count] = std::str::from_utf8(&data[begin..cursor])?.parse()?;
        count += 1;
        match data.get(cursor) {
            Some(b',') => cursor += 1,
            Some(b']') => return Ok(count),
            _ => return Err("invalid token_ids delimiter".into()),
        }
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn stable_request_id(bytes: &[u8]) -> u64 {
    let mut value = 0xcbf29ce484222325_u64;
    for byte in bytes {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0x100000001b3);
    }
    value
}

fn monotonic_ns() -> Result<u64, Box<dyn Error>> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) };
    if result != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let seconds = u64::try_from(value.tv_sec)?;
    let nanos = u64::try_from(value.tv_nsec)?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|base| base.checked_add(nanos))
        .ok_or_else(|| "CLOCK_MONOTONIC overflow".into())
}

const _: () = assert!(std::mem::size_of::<SemanticWireRecord>() == SEMANTIC_RECORD_BYTES);
