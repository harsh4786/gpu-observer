use std::ffi::CStr;
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::ptr;
use std::sync::atomic::Ordering;

use gpu_observer_core::{
    SemanticRecordFlags, SemanticRingHeader, SemanticWireRecord, SEMANTIC_RECORD_BYTES,
    SEMANTIC_RING_HEADER_BYTES,
};

const MIN_CAPACITY: u32 = 64;
const MAX_CAPACITY: u32 = 1 << 20;
const EMIT_PUBLISHED: i32 = 0;
const EMIT_DROPPED: i32 = 1;
const EMIT_INVALID: i32 = -1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SemanticSliceInput {
    pub request_id: u64,
    pub sequence_id: u64,
    pub scheduled_tokens: u32,
    pub service_class_id: u16,
    pub phase: u8,
    pub reserved: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedSliceInput {
    pub request_id: u64,
    pub packing_generation: u64,
    pub row_begin: u32,
    pub row_end: u32,
    pub scheduled_tokens: u32,
    pub packed_index: u32,
    pub phase: u8,
    pub reserved: [u8; 7],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedTokenInput {
    pub request_id: u64,
    pub packing_generation: u64,
    pub packed_row: u32,
    pub sequence_position: u32,
    pub token_id: u32,
    pub phase: u8,
    pub reserved: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OutputTokenInput {
    pub request_id: u64,
    pub output_position: u32,
    pub token_id: u32,
}

pub struct SemanticBridge {
    mapping: *mut u8,
    mapping_len: usize,
    header: *mut SemanticRingHeader,
    records: *mut SemanticWireRecord,
    capacity: u64,
    mask: u64,
    head: u64,
    next_sequence: u64,
    dropped_pending: bool,
    pid: u32,
    tid: u32,
}

impl SemanticBridge {
    fn open(path: &Path, capacity: u32) -> Option<Self> {
        if !(MIN_CAPACITY..=MAX_CAPACITY).contains(&capacity) || !capacity.is_power_of_two() {
            return None;
        }
        let records_len = usize::try_from(capacity)
            .ok()?
            .checked_mul(SEMANTIC_RECORD_BYTES)?;
        let mapping_len = SEMANTIC_RING_HEADER_BYTES.checked_add(records_len)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .ok()?;
        file.set_len(u64::try_from(mapping_len).ok()?).ok()?;

        let mapping = unsafe {
            libc::mmap(
                ptr::null_mut(),
                mapping_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return None;
        }

        let mapping = mapping.cast::<u8>();
        let header = mapping.cast::<SemanticRingHeader>();
        let records = unsafe {
            mapping
                .add(SEMANTIC_RING_HEADER_BYTES)
                .cast::<SemanticWireRecord>()
        };
        unsafe {
            ptr::write(header, SemanticRingHeader::new(capacity));
        }

        Some(Self {
            mapping,
            mapping_len,
            header,
            records,
            capacity: u64::from(capacity),
            mask: u64::from(capacity - 1),
            head: 0,
            next_sequence: 0,
            dropped_pending: false,
            pid: unsafe { libc::getpid() as u32 },
            tid: unsafe { libc::syscall(libc::SYS_gettid) as u32 },
        })
    }

    #[inline]
    fn header(&self) -> &SemanticRingHeader {
        unsafe { &*self.header }
    }

    #[inline]
    fn reserve(&mut self, count: u64) -> Option<(u64, u32)> {
        if count == 0 || count >= self.capacity {
            return None;
        }
        let tail = self.header().load_tail();
        if self.head.wrapping_sub(tail) > self.capacity.saturating_sub(count) {
            self.next_sequence = self.next_sequence.wrapping_add(count);
            self.dropped_pending = true;
            self.header().dropped.fetch_add(count, Ordering::Relaxed);
            return None;
        }
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(count);
        let flags = if self.dropped_pending {
            SemanticRecordFlags::DROPPED_BEFORE
        } else {
            0
        };
        Some((sequence, flags))
    }

    #[inline]
    unsafe fn write_at(&mut self, offset: u64, record: SemanticWireRecord) {
        let index = self.head.wrapping_add(offset) & self.mask;
        unsafe {
            ptr::write(self.records.add(index as usize), record);
        }
    }

    #[inline]
    fn publish(&mut self, count: u64) {
        self.head = self.head.wrapping_add(count);
        self.header().publish_head(self.head);
        self.dropped_pending = false;
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn emit_step_begin(
        &mut self,
        timestamp_ns: u64,
        step_id: u64,
        scheduled_tokens: u32,
        prefill_tokens: u32,
        decode_tokens: u32,
        queue_depth: u32,
        active_requests: u32,
        kv_cache_usage_permyriad: u16,
        slices: *const SemanticSliceInput,
        slice_count: u32,
    ) -> i32 {
        if slice_count != 0 && slices.is_null() {
            return EMIT_INVALID;
        }
        let required = u64::from(slice_count) + 1;
        let Some((sequence, flags)) = self.reserve(required) else {
            return EMIT_DROPPED;
        };
        unsafe {
            self.write_at(
                0,
                SemanticWireRecord::step_begin(
                    timestamp_ns,
                    sequence,
                    step_id,
                    scheduled_tokens,
                    prefill_tokens,
                    decode_tokens,
                    queue_depth,
                    active_requests,
                    slice_count,
                    kv_cache_usage_permyriad,
                    self.pid,
                    self.tid,
                    flags,
                ),
            );
        }

        let inputs = if slice_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(slices, slice_count as usize) }
        };
        for (index, input) in inputs.iter().enumerate() {
            unsafe {
                self.write_at(
                    index as u64 + 1,
                    SemanticWireRecord::request_slice(
                        timestamp_ns,
                        sequence.wrapping_add(index as u64 + 1),
                        step_id,
                        input.request_id,
                        input.sequence_id,
                        input.scheduled_tokens,
                        input.service_class_id,
                        input.phase,
                        self.pid,
                        self.tid,
                    ),
                );
            }
        }
        self.publish(required);
        EMIT_PUBLISHED
    }

    unsafe fn emit_packed_layout(
        &mut self,
        timestamp_ns: u64,
        step_id: u64,
        packing_generation: u64,
        total_tokens: u32,
        slices: *const PackedSliceInput,
        slice_count: u32,
    ) -> i32 {
        if step_id == 0 || packing_generation == 0 || slice_count == 0 || slices.is_null() {
            return EMIT_INVALID;
        }
        let inputs = unsafe { std::slice::from_raw_parts(slices, slice_count as usize) };
        let mut expected_begin = 0_u32;
        for (index, input) in inputs.iter().enumerate() {
            if input.packing_generation != packing_generation
                || input.packed_index as usize != index
                || input.row_begin != expected_begin
                || input.row_end < input.row_begin
                || input.row_end - input.row_begin != input.scheduled_tokens
                || input.scheduled_tokens == 0
            {
                return EMIT_INVALID;
            }
            expected_begin = input.row_end;
        }
        if expected_begin != total_tokens {
            return EMIT_INVALID;
        }

        let required = u64::from(slice_count) + 1;
        let Some((sequence, flags)) = self.reserve(required) else {
            return EMIT_DROPPED;
        };
        unsafe {
            self.write_at(
                0,
                SemanticWireRecord::packed_layout_begin(
                    timestamp_ns,
                    sequence,
                    step_id,
                    packing_generation,
                    total_tokens,
                    slice_count,
                    self.pid,
                    self.tid,
                    flags,
                ),
            );
        }
        for (index, input) in inputs.iter().enumerate() {
            unsafe {
                self.write_at(
                    index as u64 + 1,
                    SemanticWireRecord::packed_request_slice(
                        timestamp_ns,
                        sequence.wrapping_add(index as u64 + 1),
                        step_id,
                        input.request_id,
                        packing_generation,
                        input.row_begin,
                        input.row_end,
                        input.scheduled_tokens,
                        input.packed_index,
                        input.phase,
                        self.pid,
                        self.tid,
                    ),
                );
            }
        }
        self.publish(required);
        EMIT_PUBLISHED
    }

    unsafe fn emit_packed_tokens(
        &mut self,
        timestamp_ns: u64,
        step_id: u64,
        packing_generation: u64,
        tokens: *const PackedTokenInput,
        token_count: u32,
    ) -> i32 {
        if step_id == 0 || packing_generation == 0 || token_count == 0 || tokens.is_null() {
            return EMIT_INVALID;
        }
        let inputs = unsafe { std::slice::from_raw_parts(tokens, token_count as usize) };
        for (index, input) in inputs.iter().enumerate() {
            if input.request_id == 0
                || input.packing_generation != packing_generation
                || !matches!(input.phase, 1 | 2)
                || (index != 0
                    && (input.packed_row != inputs[index - 1].packed_row + 1
                        || input.sequence_position != inputs[index - 1].sequence_position + 1))
            {
                return EMIT_INVALID;
            }
        }

        let required = u64::from(token_count);
        let Some((sequence, flags)) = self.reserve(required) else {
            return EMIT_DROPPED;
        };
        for (index, input) in inputs.iter().enumerate() {
            unsafe {
                self.write_at(
                    index as u64,
                    SemanticWireRecord::packed_token_row(
                        timestamp_ns,
                        sequence.wrapping_add(index as u64),
                        step_id,
                        input.request_id,
                        packing_generation,
                        input.packed_row,
                        input.sequence_position,
                        input.token_id,
                        input.phase,
                        self.pid,
                        self.tid,
                        if index == 0 { flags } else { 0 },
                    ),
                );
            }
        }
        self.publish(required);
        EMIT_PUBLISHED
    }

    unsafe fn emit_output_tokens(
        &mut self,
        timestamp_ns: u64,
        step_id: u64,
        tokens: *const OutputTokenInput,
        token_count: u32,
    ) -> i32 {
        if step_id == 0 || token_count == 0 || tokens.is_null() {
            return EMIT_INVALID;
        }
        let inputs = unsafe { std::slice::from_raw_parts(tokens, token_count as usize) };
        for (index, input) in inputs.iter().enumerate() {
            if input.request_id == 0
                || (index != 0 && input.output_position != inputs[index - 1].output_position + 1)
            {
                return EMIT_INVALID;
            }
        }

        let required = u64::from(token_count);
        let Some((sequence, flags)) = self.reserve(required) else {
            return EMIT_DROPPED;
        };
        for (index, input) in inputs.iter().enumerate() {
            unsafe {
                self.write_at(
                    index as u64,
                    SemanticWireRecord::accepted_output_token(
                        timestamp_ns,
                        sequence.wrapping_add(index as u64),
                        step_id,
                        input.request_id,
                        input.output_position,
                        input.token_id,
                        self.pid,
                        self.tid,
                        if index == 0 { flags } else { 0 },
                    ),
                );
            }
        }
        self.publish(required);
        EMIT_PUBLISHED
    }

    fn emit_step_end(&mut self, timestamp_ns: u64, step_id: u64, status: u8) -> i32 {
        let Some((sequence, flags)) = self.reserve(1) else {
            return EMIT_DROPPED;
        };
        unsafe {
            self.write_at(
                0,
                SemanticWireRecord::step_end(
                    timestamp_ns,
                    sequence,
                    step_id,
                    status,
                    self.pid,
                    self.tid,
                    flags,
                ),
            );
        }
        self.publish(1);
        EMIT_PUBLISHED
    }
}

impl Drop for SemanticBridge {
    fn drop(&mut self) {
        unsafe {
            libc::msync(self.mapping.cast(), self.mapping_len, libc::MS_ASYNC);
            libc::munmap(self.mapping.cast(), self.mapping_len);
        }
    }
}

/// A lightweight, non-destructive handle onto an *existing* semantic ring's
/// shared header, used to set the live focus request hash from a process other
/// than the one that created the ring (the vLLM frontend process, which admits
/// a request and knows its internal ID before EngineCore's scheduler ever sees
/// it). Unlike `SemanticBridge::open`, this never creates, truncates, or
/// resizes the backing file: the ring must already have been created by its
/// producer. Opening it with `O_CREAT|O_TRUNC` from a second process would
/// reset head/tail/dropped and the record array out from under the producer
/// and any live reader — this type exists specifically to avoid that.
pub struct SemanticFocusHandle {
    mapping: *mut u8,
    mapping_len: usize,
    header: *mut SemanticRingHeader,
}

impl SemanticFocusHandle {
    fn open(path: &Path) -> Option<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path).ok()?;
        let mapping_len = SEMANTIC_RING_HEADER_BYTES;
        if file.metadata().ok()?.len() < mapping_len as u64 {
            return None;
        }
        let mapping = unsafe {
            libc::mmap(
                ptr::null_mut(),
                mapping_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return None;
        }
        let mapping = mapping.cast::<u8>();
        let header = mapping.cast::<SemanticRingHeader>();
        let header_ref = unsafe { &*header };
        if header_ref.validate().is_err() {
            unsafe {
                libc::munmap(mapping.cast(), mapping_len);
            }
            return None;
        }
        Some(Self {
            mapping,
            mapping_len,
            header,
        })
    }

    #[inline]
    fn header(&self) -> &SemanticRingHeader {
        unsafe { &*self.header }
    }
}

impl Drop for SemanticFocusHandle {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mapping.cast(), self.mapping_len);
        }
    }
}

/// Opens a bounded semantic bridge.
///
/// # Safety
/// `path` must point to a readable NUL-terminated string for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_bridge_open(
    path: *const libc::c_char,
    capacity: u32,
) -> *mut SemanticBridge {
    if path.is_null() {
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(path) };
    let path = Path::new(std::ffi::OsStr::from_bytes(path.to_bytes()));
    SemanticBridge::open(path, capacity)
        .map(Box::new)
        .map_or(ptr::null_mut(), Box::into_raw)
}

/// Closes a semantic bridge returned by `gpu_observer_bridge_open`.
///
/// # Safety
/// `bridge` must be null or a live pointer returned by `gpu_observer_bridge_open`, and it
/// must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_bridge_close(bridge: *mut SemanticBridge) {
    if !bridge.is_null() {
        drop(unsafe { Box::from_raw(bridge) });
    }
}

/// Publishes one engine-step semantic group.
///
/// # Safety
/// `bridge` must be a live bridge pointer. When `slice_count` is nonzero, `slices` must
/// reference that many initialized `SemanticSliceInput` values for the duration of the call.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpu_observer_emit_step_begin(
    bridge: *mut SemanticBridge,
    timestamp_ns: u64,
    step_id: u64,
    scheduled_tokens: u32,
    prefill_tokens: u32,
    decode_tokens: u32,
    queue_depth: u32,
    active_requests: u32,
    kv_cache_usage_permyriad: u16,
    slices: *const SemanticSliceInput,
    slice_count: u32,
) -> i32 {
    let Some(bridge) = (unsafe { bridge.as_mut() }) else {
        return EMIT_INVALID;
    };
    unsafe {
        bridge.emit_step_begin(
            timestamp_ns,
            step_id,
            scheduled_tokens,
            prefill_tokens,
            decode_tokens,
            queue_depth,
            active_requests,
            kv_cache_usage_permyriad,
            slices,
            slice_count,
        )
    }
}

/// Publishes one authoritative packed-layout group.
///
/// # Safety
/// `bridge` must be live. When `slice_count` is nonzero, `slices` must reference that many
/// initialized `PackedSliceInput` values for the duration of the call.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpu_observer_emit_packed_layout(
    bridge: *mut SemanticBridge,
    timestamp_ns: u64,
    step_id: u64,
    packing_generation: u64,
    total_tokens: u32,
    slices: *const PackedSliceInput,
    slice_count: u32,
) -> i32 {
    let Some(bridge) = (unsafe { bridge.as_mut() }) else {
        return EMIT_INVALID;
    };
    unsafe {
        bridge.emit_packed_layout(
            timestamp_ns,
            step_id,
            packing_generation,
            total_tokens,
            slices,
            slice_count,
        )
    }
}

/// Publishes focused packed-token records.
///
/// # Safety
/// `bridge` must be live. `tokens` must reference `token_count` initialized
/// `PackedTokenInput` values for the duration of the call.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpu_observer_emit_packed_tokens(
    bridge: *mut SemanticBridge,
    timestamp_ns: u64,
    step_id: u64,
    packing_generation: u64,
    tokens: *const PackedTokenInput,
    token_count: u32,
) -> i32 {
    let Some(bridge) = (unsafe { bridge.as_mut() }) else {
        return EMIT_INVALID;
    };
    unsafe {
        bridge.emit_packed_tokens(
            timestamp_ns,
            step_id,
            packing_generation,
            tokens,
            token_count,
        )
    }
}

/// Publishes focused accepted-output-token records.
///
/// # Safety
/// `bridge` must be live. `tokens` must reference `token_count` initialized
/// `OutputTokenInput` values for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_emit_output_tokens(
    bridge: *mut SemanticBridge,
    timestamp_ns: u64,
    step_id: u64,
    tokens: *const OutputTokenInput,
    token_count: u32,
) -> i32 {
    let Some(bridge) = (unsafe { bridge.as_mut() }) else {
        return EMIT_INVALID;
    };
    unsafe { bridge.emit_output_tokens(timestamp_ns, step_id, tokens, token_count) }
}

/// Publishes the end of an engine step.
///
/// # Safety
/// `bridge` must be a live pointer returned by `gpu_observer_bridge_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_emit_step_end(
    bridge: *mut SemanticBridge,
    timestamp_ns: u64,
    step_id: u64,
    status: u8,
) -> i32 {
    let Some(bridge) = (unsafe { bridge.as_mut() }) else {
        return EMIT_INVALID;
    };
    bridge.emit_step_end(timestamp_ns, step_id, status)
}

/// Returns the producer-side drop counter.
///
/// # Safety
/// `bridge` must be null or a live pointer returned by `gpu_observer_bridge_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_bridge_dropped(bridge: *const SemanticBridge) -> u64 {
    let Some(bridge) = (unsafe { bridge.as_ref() }) else {
        return 0;
    };
    bridge.header().dropped.load(Ordering::Relaxed)
}

/// Returns the request hash currently in focus for per-token detail emission
/// (`packed_token_row` / `accepted_output_token`), or 0 if none is set. Read
/// live on every call site that gates on focus; not cached at bridge-open time.
///
/// # Safety
/// `bridge` must be null or a live pointer returned by `gpu_observer_bridge_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_bridge_get_focus(bridge: *const SemanticBridge) -> u64 {
    let Some(bridge) = (unsafe { bridge.as_ref() }) else {
        return 0;
    };
    bridge.header().load_focus()
}

/// Sets the request hash currently in focus, from the ring's own producer
/// process. Used to seed a static, pre-known focus at bridge-open time
/// (preserving the old single-run GPU_OBSERVER_FOCUS_REQUEST_ID capture
/// behavior). For dynamic per-request focus set from a different process,
/// use gpu_observer_bridge_focus_open/_set instead -- that path never
/// truncates or recreates the ring; this one is only safe because the
/// caller already owns the mapping it is writing into.
///
/// # Safety
/// `bridge` must be null or a live pointer returned by `gpu_observer_bridge_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_bridge_set_focus(
    bridge: *const SemanticBridge,
    request_id_hash: u64,
) -> i32 {
    let Some(bridge) = (unsafe { bridge.as_ref() }) else {
        return EMIT_INVALID;
    };
    bridge.header().store_focus(request_id_hash);
    EMIT_PUBLISHED
}

/// Opens a non-destructive focus-only handle onto an already-created semantic
/// ring, for use by a process that did not create the ring (the vLLM frontend).
/// Returns null if the ring file does not exist yet or fails header validation.
///
/// # Safety
/// `path` must point to a readable NUL-terminated string for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_bridge_focus_open(
    path: *const libc::c_char,
) -> *mut SemanticFocusHandle {
    if path.is_null() {
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(path) };
    let path = Path::new(std::ffi::OsStr::from_bytes(path.to_bytes()));
    SemanticFocusHandle::open(path)
        .map(Box::new)
        .map_or(ptr::null_mut(), Box::into_raw)
}

/// Closes a handle returned by `gpu_observer_bridge_focus_open`.
///
/// # Safety
/// `handle` must be null or a live pointer returned by `gpu_observer_bridge_focus_open`,
/// and it must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_bridge_focus_close(handle: *mut SemanticFocusHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Sets the request hash currently in focus for per-token detail emission.
/// A single `Release` store to the ring's focus cursor; never touches
/// head/tail/dropped or the record array, so it is safe to call from a
/// different process than the one that created the ring, concurrently with
/// the producer emitting records and a reader draining them.
///
/// # Safety
/// `handle` must be null or a live pointer returned by `gpu_observer_bridge_focus_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpu_observer_bridge_focus_set(
    handle: *const SemanticFocusHandle,
    request_id_hash: u64,
) -> i32 {
    let Some(handle) = (unsafe { handle.as_ref() }) else {
        return EMIT_INVALID;
    };
    handle.header().store_focus(request_id_hash);
    EMIT_PUBLISHED
}

const _: () = assert!(std::mem::size_of::<SemanticSliceInput>() == 24);
const _: () = assert!(std::mem::size_of::<PackedTokenInput>() == 32);
const _: () = assert!(std::mem::size_of::<OutputTokenInput>() == 16);
const _: () = assert!(std::mem::size_of::<PackedSliceInput>() == 40);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_slice_abi_is_fixed() {
        assert_eq!(std::mem::size_of::<SemanticSliceInput>(), 24);
        assert_eq!(std::mem::align_of::<SemanticSliceInput>(), 8);
        assert_eq!(std::mem::size_of::<PackedSliceInput>(), 40);
        assert_eq!(std::mem::size_of::<PackedTokenInput>(), 32);
        assert_eq!(std::mem::align_of::<PackedTokenInput>(), 8);
        assert_eq!(std::mem::size_of::<OutputTokenInput>(), 16);
        assert_eq!(std::mem::align_of::<OutputTokenInput>(), 8);
        assert_eq!(std::mem::align_of::<PackedSliceInput>(), 8);
    }
}
