use std::env;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};
use std::time::Duration;

const FILE_MAGIC: &[u8; 8] = b"GOQRY01\0";
const FILE_VERSION: u32 = 1;
const MAX_DATAGRAM_BYTES: usize = 1024 * 1024;

struct SocketGuard(PathBuf);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn parse_positive<T>(value: &str, name: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr + PartialOrd + From<u8>,
    T::Err: std::error::Error + 'static,
{
    let parsed = value.parse::<T>()?;
    if parsed <= T::from(0) {
        return Err(format!("{name} must be positive").into());
    }
    Ok(parsed)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "query_capture".to_owned());
    let usage = || format!("usage: {program} SOCKET OUTPUT.msgpack.frames MAX_FRAMES TIMEOUT_MS");
    let socket_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let output_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let max_frames: u32 = parse_positive(&args.next().ok_or_else(&usage)?, "MAX_FRAMES")?;
    let timeout_ms: u64 = parse_positive(&args.next().ok_or_else(&usage)?, "TIMEOUT_MS")?;
    if args.next().is_some() {
        return Err(usage().into());
    }
    if socket_path.exists() {
        return Err(format!("refusing existing socket path: {}", socket_path.display()).into());
    }
    if output_path.exists() {
        return Err(format!("refusing existing output path: {}", output_path.display()).into());
    }
    if let Some(parent) = socket_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            return Err(format!("socket parent does not exist: {}", parent.display()).into());
        }
    }

    let socket = UnixDatagram::bind(&socket_path)?;
    let _socket_guard = SocketGuard(socket_path.clone());
    socket.set_read_timeout(Some(Duration::from_millis(timeout_ms)))?;

    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output_path)?;
    output.write_all(FILE_MAGIC)?;
    output.write_all(&FILE_VERSION.to_le_bytes())?;
    output.write_all(&(MAX_DATAGRAM_BYTES as u32).to_le_bytes())?;
    output.write_all(&0_u64.to_le_bytes())?;

    let mut buffer = Vec::new();
    buffer.try_reserve_exact(MAX_DATAGRAM_BYTES)?;
    buffer.resize(MAX_DATAGRAM_BYTES, 0);
    let mut frames = 0_u32;
    loop {
        match socket.recv(&mut buffer) {
            Ok(bytes) => {
                if bytes == 0 || bytes > MAX_DATAGRAM_BYTES {
                    return Err("received invalid query manifest length".into());
                }
                output.write_all(&(bytes as u32).to_le_bytes())?;
                output.write_all(&buffer[..bytes])?;
                frames += 1;
                if frames == max_frames {
                    break;
                }
            }
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                if frames == 0 {
                    return Err("timed out before receiving a query manifest".into());
                }
                break;
            }
            Err(error) => return Err(error.into()),
        }
    }

    output.flush()?;
    output.sync_data()?;
    println!(
        "query_capture frames={} bytes={} socket={} output={}",
        frames,
        Path::new(&output_path).metadata()?.len(),
        socket_path.display(),
        output_path.display()
    );
    Ok(())
}
