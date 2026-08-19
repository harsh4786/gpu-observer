#![no_std]

use core::{mem::size_of, ptr};

pub const CUDA_LAUNCH_MAGIC: u32 = u32::from_le_bytes(*b"GPOB");
pub const CUDA_LAUNCH_ABI_VERSION: u16 = 1;
pub const CUDA_LAUNCH_EVENT_BYTES: usize = 88;

pub struct LaunchFlags;

impl LaunchFlags {
    pub const STREAM_VALID: u32 = 1 << 0;
    pub const DROPPED_BEFORE: u32 = 1 << 1;
    /// Launch entered through `cuLaunchKernelEx` and used `CUlaunchConfig`.
    pub const EXTENDED_CONFIG: u32 = 1 << 2;
}

/// Integer-only wire format shared by the eBPF producer and Rust loader.
///
/// All bytes have valid bit patterns. The loader validates magic, version,
/// and size before converting this record into the internal typed schema.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaLaunchEvent {
    pub timestamp_ns: u64,
    /// Monotonic attempt sequence local to `cpu_id`; gaps mean ring loss.
    pub sequence: u64,
    pub kernel_function: u64,
    pub stream: u64,
    pub pid: u32,
    pub tid: u32,
    pub cpu_id: u32,
    pub shared_memory_bytes: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub magic: u32,
    pub flags: u32,
    pub abi_version: u16,
    pub record_size: u16,
    pub reserved: u32,
}

impl CudaLaunchEvent {
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub const fn new(
        timestamp_ns: u64,
        sequence: u64,
        kernel_function: u64,
        stream: u64,
        pid: u32,
        tid: u32,
        cpu_id: u32,
        shared_memory_bytes: u32,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        flags: u32,
    ) -> Self {
        Self {
            timestamp_ns,
            sequence,
            kernel_function,
            stream,
            pid,
            tid,
            cpu_id,
            shared_memory_bytes,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            magic: CUDA_LAUNCH_MAGIC,
            flags,
            abi_version: CUDA_LAUNCH_ABI_VERSION,
            record_size: CUDA_LAUNCH_EVENT_BYTES as u16,
            reserved: 0,
        }
    }

    #[inline]
    pub const fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    /// Copies from possibly unaligned ring-buffer bytes, then validates the ABI.
    #[inline]
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() != CUDA_LAUNCH_EVENT_BYTES {
            return Err(DecodeError::Length);
        }

        // Every field is an integer, so every copied bit pattern is valid.
        let event = unsafe { ptr::read_unaligned(bytes.as_ptr().cast::<Self>()) };
        if event.magic != CUDA_LAUNCH_MAGIC {
            return Err(DecodeError::Magic);
        }
        if event.abi_version != CUDA_LAUNCH_ABI_VERSION {
            return Err(DecodeError::Version);
        }
        if usize::from(event.record_size) != CUDA_LAUNCH_EVENT_BYTES {
            return Err(DecodeError::RecordSize);
        }
        Ok(event)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    Length,
    Magic,
    Version,
    RecordSize,
}

const _: () = assert!(size_of::<CudaLaunchEvent>() == CUDA_LAUNCH_EVENT_BYTES);
const _: () = assert!(core::mem::align_of::<CudaLaunchEvent>() == 8);

#[cfg(feature = "user")]
unsafe impl aya::Pod for CudaLaunchEvent {}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_of(value: &CudaLaunchEvent) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(ptr::from_ref(value).cast::<u8>(), CUDA_LAUNCH_EVENT_BYTES)
        }
    }

    #[test]
    fn wire_layout_is_fixed_and_round_trips() {
        let event = CudaLaunchEvent::new(
            100,
            7,
            0x1111,
            0x2222,
            42,
            43,
            9,
            512,
            8,
            4,
            1,
            128,
            1,
            1,
            LaunchFlags::STREAM_VALID,
        );
        assert_eq!(size_of::<CudaLaunchEvent>(), CUDA_LAUNCH_EVENT_BYTES);
        assert_eq!(CudaLaunchEvent::decode(bytes_of(&event)), Ok(event));
    }

    #[test]
    fn decoder_rejects_wrong_length_and_version() {
        assert_eq!(CudaLaunchEvent::decode(&[0; 8]), Err(DecodeError::Length));

        let mut event = CudaLaunchEvent::default();
        event.magic = CUDA_LAUNCH_MAGIC;
        event.abi_version = CUDA_LAUNCH_ABI_VERSION + 1;
        event.record_size = CUDA_LAUNCH_EVENT_BYTES as u16;
        assert_eq!(
            CudaLaunchEvent::decode(bytes_of(&event)),
            Err(DecodeError::Version)
        );
    }
}
