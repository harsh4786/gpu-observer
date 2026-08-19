use std::{
    env,
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroU32,
    os::fd::AsRawFd,
    path::PathBuf,
    time::{Duration, Instant},
};

use aya::{
    maps::RingBuf,
    programs::{uprobe::UProbeScope, UProbe},
    Ebpf,
};
use gpu_observer_host_probe_common::{CudaLaunchEvent, LaunchFlags};

const DEFAULT_DURATION_SECONDS: u64 = 15;
const POLL_INTERVAL_MS: i32 = 10;
const SAMPLE_LIMIT: u64 = 8;

struct Options {
    pid: NonZeroU32,
    library: PathBuf,
    object: PathBuf,
    duration: Duration,
    raw_output: Option<PathBuf>,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = env::args_os();
        let program = args.next().unwrap_or_default();
        let usage = || {
            format!(
                "usage: {} <host-pid> <libcuda-path> <ebpf-object> [duration-seconds] [raw-output.bin]",
                PathBuf::from(&program).display()
            )
        };

        let pid: u32 = args.next().ok_or_else(&usage)?.to_string_lossy().parse()?;
        let pid = NonZeroU32::new(pid).ok_or("PID must be non-zero")?;
        let library = PathBuf::from(args.next().ok_or_else(&usage)?);
        let object = PathBuf::from(args.next().ok_or_else(&usage)?);
        let duration = match args.next() {
            Some(value) => Duration::from_secs(value.to_string_lossy().parse()?),
            None => Duration::from_secs(DEFAULT_DURATION_SECONDS),
        };
        let raw_output = args.next().map(PathBuf::from);
        if args.next().is_some() {
            return Err(usage().into());
        }

        Ok(Self {
            pid,
            library,
            object,
            duration,
            raw_output,
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    raise_memlock_limit();

    let mut ebpf = Ebpf::load_file(&options.object)?;
    let program: &mut UProbe = ebpf
        .program_mut("observe_cu_launch_kernel")
        .ok_or("eBPF program observe_cu_launch_kernel is missing")?
        .try_into()?;
    program.load()?;
    program.attach(
        "cuLaunchKernel",
        &options.library,
        UProbeScope::OneProcess(options.pid),
    )?;
    let extended_program: &mut UProbe = ebpf
        .program_mut("observe_cu_launch_kernel_ex")
        .ok_or("eBPF program observe_cu_launch_kernel_ex is missing")?
        .try_into()?;
    extended_program.load()?;
    extended_program.attach(
        "cuLaunchKernelEx",
        &options.library,
        UProbeScope::OneProcess(options.pid),
    )?;

    let map = ebpf
        .take_map("EVENTS")
        .ok_or("eBPF map EVENTS is missing")?;
    let mut ring = RingBuf::try_from(map)?;
    let mut poll_fd = libc::pollfd {
        fd: ring.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let mut raw_output = options
        .raw_output
        .as_ref()
        .map(File::create)
        .transpose()?
        .map(|file| BufWriter::with_capacity(1024 * 1024, file));

    eprintln!(
        "attached pid={} symbols=cuLaunchKernel,cuLaunchKernelEx ring=8MiB duration={}s",
        options.pid,
        options.duration.as_secs()
    );

    let deadline = Instant::now() + options.duration;
    let mut events = 0_u64;
    let mut malformed = 0_u64;
    let mut loss_markers = 0_u64;
    let mut regular_events = 0_u64;
    let mut extended_events = 0_u64;

    while Instant::now() < deadline {
        // The producer suppresses per-event wakeups; a short bounded poll keeps
        // submission threads free of wakeup work while bounding drain latency.
        unsafe {
            libc::poll(&mut poll_fd, 1, POLL_INTERVAL_MS);
        }
        poll_fd.revents = 0;

        while let Some(bytes) = ring.next() {
            if let Some(output) = raw_output.as_mut() {
                output.write_all(&bytes)?;
            }
            let event = match CudaLaunchEvent::decode(&bytes) {
                Ok(event) => event,
                Err(_) => {
                    malformed = malformed.saturating_add(1);
                    continue;
                }
            };
            events = events.saturating_add(1);
            if event.has_flag(LaunchFlags::EXTENDED_CONFIG) {
                extended_events = extended_events.saturating_add(1);
            } else {
                regular_events = regular_events.saturating_add(1);
            }
            if event.has_flag(LaunchFlags::DROPPED_BEFORE) {
                loss_markers = loss_markers.saturating_add(1);
            }
            if events <= SAMPLE_LIMIT {
                let api = if event.has_flag(LaunchFlags::EXTENDED_CONFIG) {
                    "cuLaunchKernelEx"
                } else {
                    "cuLaunchKernel"
                };
                println!(
                    "ts={} cpu={} pid={} tid={} seq={} api={} fn=0x{:x} stream=0x{:x} grid=({},{},{}) block=({},{},{}) smem={} flags=0x{:x}",
                    event.timestamp_ns,
                    event.cpu_id,
                    event.pid,
                    event.tid,
                    event.sequence,
                    api,
                    event.kernel_function,
                    event.stream,
                    event.grid_x,
                    event.grid_y,
                    event.grid_z,
                    event.block_x,
                    event.block_y,
                    event.block_z,
                    event.shared_memory_bytes,
                    event.flags,
                );
            }
        }
    }

    if let Some(output) = raw_output.as_mut() {
        output.flush()?;
        output.get_ref().sync_all()?;
    }

    println!(
        "summary events={} regular={} extended={} malformed={} loss_markers={}",
        events, regular_events, extended_events, malformed, loss_markers
    );
    Ok(())
}

fn raise_memlock_limit() {
    let limit = libc::rlimit {
        rlim_cur: libc::RLIM_INFINITY,
        rlim_max: libc::RLIM_INFINITY,
    };
    unsafe {
        libc::setrlimit(libc::RLIMIT_MEMLOCK, &limit);
    }
}
