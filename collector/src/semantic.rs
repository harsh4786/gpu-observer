use std::fs::OpenOptions;
use std::io::{Error, ErrorKind};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;

use gpu_observer_core::{
    SemanticRingHeader, SemanticWireRecord, SEMANTIC_RECORD_BYTES, SEMANTIC_RING_HEADER_BYTES,
};

use crate::error::Result;

pub struct SemanticRingReader {
    mapping: *mut u8,
    mapping_len: usize,
    header: *mut SemanticRingHeader,
    records: *const SemanticWireRecord,
    capacity: u64,
    mask: u64,
    tail: u64,
}

impl SemanticRingReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mapping_len = usize::try_from(file.metadata()?.len())
            .map_err(|_| invalid_data("semantic ring is too large"))?;
        if mapping_len < SEMANTIC_RING_HEADER_BYTES {
            return Err(invalid_data("semantic ring header is truncated").into());
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
            return Err(Error::last_os_error().into());
        }
        let mapping = mapping.cast::<u8>();
        let header = mapping.cast::<SemanticRingHeader>();
        let header_ref = unsafe { &*header };
        if let Err(error) = header_ref.validate() {
            unsafe {
                libc::munmap(mapping.cast(), mapping_len);
            }
            return Err(invalid_data(&format!("invalid semantic ring header: {error:?}")).into());
        }

        let capacity = u64::from(header_ref.capacity);
        let expected = SEMANTIC_RING_HEADER_BYTES
            .checked_add(
                usize::try_from(capacity)
                    .ok()
                    .and_then(|value| value.checked_mul(SEMANTIC_RECORD_BYTES))
                    .ok_or_else(|| invalid_data("semantic ring length overflow"))?,
            )
            .ok_or_else(|| invalid_data("semantic ring length overflow"))?;
        if mapping_len != expected {
            unsafe {
                libc::munmap(mapping.cast(), mapping_len);
            }
            return Err(invalid_data("semantic ring file size does not match header").into());
        }

        let records = unsafe {
            mapping
                .add(SEMANTIC_RING_HEADER_BYTES)
                .cast::<SemanticWireRecord>()
        };
        let tail = header_ref.load_tail();
        Ok(Self {
            mapping,
            mapping_len,
            header,
            records,
            capacity,
            mask: capacity - 1,
            tail,
        })
    }

    #[inline]
    fn header(&self) -> &SemanticRingHeader {
        unsafe { &*self.header }
    }

    pub fn try_next(&mut self) -> Result<Option<SemanticWireRecord>> {
        let head = self.header().load_head();
        if self.tail == head {
            return Ok(None);
        }
        if head.wrapping_sub(self.tail) > self.capacity {
            return Err(invalid_data("semantic producer overran consumer cursor").into());
        }

        let index = self.tail & self.mask;
        let record = unsafe { ptr::read(self.records.add(index as usize)) };
        if let Err(error) = record.validate() {
            return Err(invalid_data(&format!("invalid semantic record: {error:?}")).into());
        }
        self.tail = self.tail.wrapping_add(1);
        self.header().publish_tail(self.tail);
        Ok(Some(record))
    }

    pub fn dropped_records(&self) -> u64 {
        self.header()
            .dropped
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for SemanticRingReader {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mapping.cast(), self.mapping_len);
        }
    }
}

fn invalid_data(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_owned())
}
