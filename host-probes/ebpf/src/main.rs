#![no_std]
#![no_main]

use aya_ebpf::{
    helpers::{
        bpf_get_current_pid_tgid, bpf_probe_read_user,
        generated::{bpf_get_smp_processor_id, bpf_ktime_get_ns},
    },
    macros::{map, uprobe},
    maps::{PerCpuArray, RingBuf},
    programs::ProbeContext,
};
use gpu_observer_host_probe_common::{CudaLaunchEvent, LaunchFlags};

const RING_BYTES: u32 = 8 * 1024 * 1024;
const BPF_RB_NO_WAKEUP: u64 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct CpuState {
    next_sequence: u64,
    dropped_pending: u64,
}

/// CUDA 13 `CUlaunchConfig` layout on the DGX Spark's AArch64 userspace ABI.
/// Keep this integer-only: the probe reads one bounded copy from user memory.
#[repr(C)]
#[derive(Clone, Copy)]
struct CuLaunchConfig {
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_memory_bytes: u32,
    _padding: u32,
    stream: u64,
    _attributes: u64,
    _num_attributes: u32,
    _tail_padding: u32,
}

#[map]
static EVENTS: RingBuf = RingBuf::with_byte_size(RING_BYTES, 0);

#[map]
static CPU_STATE: PerCpuArray<CpuState> = PerCpuArray::with_max_entries(1, 0);

#[uprobe]
pub fn observe_cu_launch_kernel(ctx: ProbeContext) -> u32 {
    match try_observe_cu_launch_kernel(ctx) {
        Ok(()) => 0,
        Err(()) => 0,
    }
}

#[inline(always)]
fn try_observe_cu_launch_kernel(ctx: ProbeContext) -> Result<(), ()> {
    let kernel_function = ctx.arg::<u64>(0).ok_or(())?;
    let grid_x = ctx.arg::<u32>(1).ok_or(())?;
    let grid_y = ctx.arg::<u32>(2).ok_or(())?;
    let grid_z = ctx.arg::<u32>(3).ok_or(())?;
    let block_x = ctx.arg::<u32>(4).ok_or(())?;
    let block_y = ctx.arg::<u32>(5).ok_or(())?;
    let block_z = ctx.arg::<u32>(6).ok_or(())?;
    let shared_memory_bytes = ctx.arg::<u32>(7).ok_or(())?;

    let (stream, stream_valid) = read_stream(&ctx);
    let pid_tgid = bpf_get_current_pid_tgid();
    let cpu_id = unsafe { bpf_get_smp_processor_id() };
    let timestamp_ns = unsafe { bpf_ktime_get_ns() };

    let state = unsafe { &mut *CPU_STATE.get_ptr_mut(0).ok_or(())? };
    let sequence = state.next_sequence;
    state.next_sequence = state.next_sequence.wrapping_add(1);

    let mut flags = if stream_valid {
        LaunchFlags::STREAM_VALID
    } else {
        0
    };
    if state.dropped_pending != 0 {
        flags |= LaunchFlags::DROPPED_BEFORE;
    }

    let mut entry = match EVENTS.reserve::<CudaLaunchEvent>(0) {
        Some(entry) => entry,
        None => {
            state.dropped_pending = 1;
            return Ok(());
        }
    };
    entry.write(CudaLaunchEvent::new(
        timestamp_ns,
        sequence,
        kernel_function,
        stream,
        (pid_tgid >> 32) as u32,
        pid_tgid as u32,
        cpu_id,
        shared_memory_bytes,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        flags,
    ));
    entry.submit(BPF_RB_NO_WAKEUP);
    state.dropped_pending = 0;
    Ok(())
}

#[uprobe]
pub fn observe_cu_launch_kernel_ex(ctx: ProbeContext) -> u32 {
    match try_observe_cu_launch_kernel_ex(ctx) {
        Ok(()) => 0,
        Err(()) => 0,
    }
}

#[inline(always)]
fn try_observe_cu_launch_kernel_ex(ctx: ProbeContext) -> Result<(), ()> {
    let config_address = ctx.arg::<u64>(0).ok_or(())?;
    let kernel_function = ctx.arg::<u64>(1).ok_or(())?;
    let config =
        unsafe { bpf_probe_read_user(config_address as *const CuLaunchConfig).map_err(|_| ())? };
    let pid_tgid = bpf_get_current_pid_tgid();
    let cpu_id = unsafe { bpf_get_smp_processor_id() };
    let timestamp_ns = unsafe { bpf_ktime_get_ns() };

    let state = unsafe { &mut *CPU_STATE.get_ptr_mut(0).ok_or(())? };
    let sequence = state.next_sequence;
    state.next_sequence = state.next_sequence.wrapping_add(1);

    let mut flags = LaunchFlags::STREAM_VALID | LaunchFlags::EXTENDED_CONFIG;
    if state.dropped_pending != 0 {
        flags |= LaunchFlags::DROPPED_BEFORE;
    }

    let mut entry = match EVENTS.reserve::<CudaLaunchEvent>(0) {
        Some(entry) => entry,
        None => {
            state.dropped_pending = 1;
            return Ok(());
        }
    };
    entry.write(CudaLaunchEvent::new(
        timestamp_ns,
        sequence,
        kernel_function,
        config.stream,
        (pid_tgid >> 32) as u32,
        pid_tgid as u32,
        cpu_id,
        config.shared_memory_bytes,
        config.grid_x,
        config.grid_y,
        config.grid_z,
        config.block_x,
        config.block_y,
        config.block_z,
        flags,
    ));
    entry.submit(BPF_RB_NO_WAKEUP);
    state.dropped_pending = 0;
    Ok(())
}

const _: () = assert!(core::mem::size_of::<CuLaunchConfig>() == 56);

#[cfg(bpf_target_arch = "aarch64")]
#[inline(always)]
fn read_stream(ctx: &ProbeContext) -> (u64, bool) {
    // AArch64 passes the first eight integer arguments in x0..x7. CUstream is
    // cuLaunchKernel argument nine, so it is the first 64-bit stack argument.
    let stack_pointer = unsafe { (*ctx.regs).sp } as *const u64;
    match unsafe { bpf_probe_read_user(stack_pointer) } {
        Ok(stream) => (stream, true),
        Err(_) => (0, false),
    }
}

#[cfg(not(bpf_target_arch = "aarch64"))]
#[inline(always)]
fn read_stream(ctx: &ProbeContext) -> (u64, bool) {
    match ctx.arg::<u64>(8) {
        Some(stream) => (stream, true),
        None => (0, false),
    }
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[unsafe(link_section = "license")]
#[unsafe(no_mangle)]
static LICENSE: [u8; 13] = *b"Dual MIT/GPL\0";
