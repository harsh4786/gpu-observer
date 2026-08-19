use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::{Cell, UnsafeCell};
use core::fmt::{Display, Formatter};
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::sync::atomic::{AtomicUsize, Ordering};

const CACHE_LINE_BYTES: usize = 64;

#[repr(align(64))]
struct CachePadded<T>(T);

struct Slot<T> {
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Slot<T> {
    #[inline]
    const fn uninitialized() -> Self {
        Self {
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

unsafe impl<T: Send> Sync for Slot<T> {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RingError {
    CapacityMustBePowerOfTwo,
    AllocationFailed,
}

impl Display for RingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CapacityMustBePowerOfTwo => {
                formatter.write_str("ring capacity must be a power of two greater than one")
            }
            Self::AllocationFailed => formatter.write_str("unable to allocate ring storage"),
        }
    }
}

pub struct SpscRing<T> {
    slots: Box<[Slot<T>]>,
    mask: usize,
    capacity: usize,
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
}

impl<T> SpscRing<T> {
    pub fn try_new(capacity: usize) -> Result<Self, RingError> {
        if capacity < 2 || !capacity.is_power_of_two() {
            return Err(RingError::CapacityMustBePowerOfTwo);
        }

        let mut slots = Vec::new();
        slots
            .try_reserve_exact(capacity)
            .map_err(|_| RingError::AllocationFailed)?;
        for _ in 0..capacity {
            slots.push(Slot::uninitialized());
        }

        Ok(Self {
            slots: slots.into_boxed_slice(),
            mask: capacity - 1,
            capacity,
            head: CachePadded(AtomicUsize::new(0)),
            tail: CachePadded(AtomicUsize::new(0)),
        })
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn len_approx(&self) -> usize {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Acquire);
        head.wrapping_sub(tail).min(self.capacity)
    }

    pub fn split(&mut self) -> (Producer<'_, T>, Consumer<'_, T>) {
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Relaxed);
        let shared: &SpscRing<T> = self;
        (
            Producer {
                ring: shared,
                head,
                cached_tail: tail,
                not_sync: PhantomData,
            },
            Consumer {
                ring: shared,
                tail,
                cached_head: head,
                not_sync: PhantomData,
            },
        )
    }
}

impl<T> Drop for SpscRing<T> {
    fn drop(&mut self) {
        let head = self.head.0.load(Ordering::Relaxed);
        let mut tail = self.tail.0.load(Ordering::Relaxed);
        while tail != head {
            let index = tail & self.mask;
            unsafe {
                (*self.slots[index].value.get()).assume_init_drop();
            }
            tail = tail.wrapping_add(1);
        }
    }
}

pub struct Producer<'a, T> {
    ring: &'a SpscRing<T>,
    head: usize,
    cached_tail: usize,
    not_sync: PhantomData<Cell<()>>,
}

impl<T> Producer<'_, T> {
    #[inline]
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.head.wrapping_sub(self.cached_tail) >= self.ring.capacity {
            self.cached_tail = self.ring.tail.0.load(Ordering::Acquire);
            if self.head.wrapping_sub(self.cached_tail) >= self.ring.capacity {
                return Err(value);
            }
        }

        let index = self.head & self.ring.mask;
        unsafe {
            (*self.ring.slots[index].value.get()).write(value);
        }
        self.head = self.head.wrapping_add(1);
        self.ring.head.0.store(self.head, Ordering::Release);
        Ok(())
    }

    #[inline]
    pub fn remaining_approx(&mut self) -> usize {
        self.cached_tail = self.ring.tail.0.load(Ordering::Acquire);
        self.ring
            .capacity
            .saturating_sub(self.head.wrapping_sub(self.cached_tail))
    }
}

impl<T: Copy> Producer<'_, T> {
    pub fn push_slice(&mut self, values: &[T]) -> usize {
        if values.is_empty() {
            return 0;
        }
        self.cached_tail = self.ring.tail.0.load(Ordering::Acquire);
        let available = self
            .ring
            .capacity
            .saturating_sub(self.head.wrapping_sub(self.cached_tail));
        let count = available.min(values.len());

        for (offset, value) in values.iter().take(count).enumerate() {
            let index = self.head.wrapping_add(offset) & self.ring.mask;
            unsafe {
                (*self.ring.slots[index].value.get()).write(*value);
            }
        }
        self.head = self.head.wrapping_add(count);
        if count != 0 {
            self.ring.head.0.store(self.head, Ordering::Release);
        }
        count
    }

    pub fn try_push_slice_all(&mut self, values: &[T]) -> bool {
        if values.is_empty() {
            return true;
        }
        self.cached_tail = self.ring.tail.0.load(Ordering::Acquire);
        let available = self
            .ring
            .capacity
            .saturating_sub(self.head.wrapping_sub(self.cached_tail));
        if available < values.len() {
            return false;
        }

        for (offset, value) in values.iter().enumerate() {
            let index = self.head.wrapping_add(offset) & self.ring.mask;
            unsafe {
                (*self.ring.slots[index].value.get()).write(*value);
            }
        }
        self.head = self.head.wrapping_add(values.len());
        self.ring.head.0.store(self.head, Ordering::Release);
        true
    }
}

pub struct Consumer<'a, T> {
    ring: &'a SpscRing<T>,
    tail: usize,
    cached_head: usize,
    not_sync: PhantomData<Cell<()>>,
}

impl<T> Consumer<'_, T> {
    #[inline]
    pub fn try_pop(&mut self) -> Option<T> {
        if self.tail == self.cached_head {
            self.cached_head = self.ring.head.0.load(Ordering::Acquire);
            if self.tail == self.cached_head {
                return None;
            }
        }

        let index = self.tail & self.ring.mask;
        let value = unsafe { (*self.ring.slots[index].value.get()).assume_init_read() };
        self.tail = self.tail.wrapping_add(1);
        self.ring.tail.0.store(self.tail, Ordering::Release);
        Some(value)
    }

    #[inline]
    pub fn available_approx(&mut self) -> usize {
        self.cached_head = self.ring.head.0.load(Ordering::Acquire);
        self.cached_head
            .wrapping_sub(self.tail)
            .min(self.ring.capacity)
    }
}

impl<T: Copy> Consumer<'_, T> {
    pub fn pop_slice(&mut self, output: &mut [MaybeUninit<T>]) -> usize {
        if output.is_empty() {
            return 0;
        }
        self.cached_head = self.ring.head.0.load(Ordering::Acquire);
        let available = self
            .cached_head
            .wrapping_sub(self.tail)
            .min(self.ring.capacity);
        let count = available.min(output.len());

        for (offset, slot) in output.iter_mut().take(count).enumerate() {
            let index = self.tail.wrapping_add(offset) & self.ring.mask;
            let value = unsafe { (*self.ring.slots[index].value.get()).assume_init_read() };
            slot.write(value);
        }
        self.tail = self.tail.wrapping_add(count);
        if count != 0 {
            self.ring.tail.0.store(self.tail, Ordering::Release);
        }
        count
    }
}

const _: () = assert!(mem::align_of::<CachePadded<AtomicUsize>>() == CACHE_LINE_BYTES);
const _: () = assert!(mem::size_of::<CachePadded<AtomicUsize>>() == CACHE_LINE_BYTES);

#[cfg(test)]
mod tests {
    use core::mem::MaybeUninit;

    use super::{RingError, SpscRing};

    #[test]
    fn rejects_capacities_that_make_masking_invalid() {
        assert!(matches!(
            SpscRing::<u64>::try_new(3),
            Err(RingError::CapacityMustBePowerOfTwo)
        ));
    }

    #[test]
    fn full_ring_returns_ownership_to_the_producer() {
        let mut ring = SpscRing::try_new(2).unwrap();
        let (mut producer, mut consumer) = ring.split();
        assert_eq!(producer.try_push(10), Ok(()));
        assert_eq!(producer.try_push(20), Ok(()));
        assert_eq!(producer.try_push(30), Err(30));
        assert_eq!(consumer.try_pop(), Some(10));
        assert_eq!(consumer.try_pop(), Some(20));
        assert_eq!(consumer.try_pop(), None);
    }

    #[test]
    fn batched_operations_wrap_without_reordering() {
        let mut ring = SpscRing::try_new(4).unwrap();
        let (mut producer, mut consumer) = ring.split();
        assert_eq!(producer.push_slice(&[1, 2, 3]), 3);
        assert_eq!(consumer.try_pop(), Some(1));
        assert_eq!(producer.push_slice(&[4, 5, 6]), 2);

        let mut output = [MaybeUninit::uninit(); 4];
        assert_eq!(consumer.pop_slice(&mut output), 4);
        let values = output.map(|value| unsafe { value.assume_init() });
        assert_eq!(values, [2, 3, 4, 5]);
    }

    #[test]
    fn producer_and_consumer_can_run_on_separate_cores() {
        const ITEMS: usize = 200_000;
        let mut ring = SpscRing::try_new(1 << 12).unwrap();
        let (mut producer, mut consumer) = ring.split();

        std::thread::scope(|scope| {
            let produced = scope.spawn(move || {
                for value in 0..ITEMS {
                    let mut pending = value as u64;
                    loop {
                        match producer.try_push(pending) {
                            Ok(()) => break,
                            Err(value) => {
                                pending = value;
                                core::hint::spin_loop();
                            }
                        }
                    }
                }
            });
            let consumed = scope.spawn(move || {
                let mut expected = 0_u64;
                while expected < ITEMS as u64 {
                    if let Some(value) = consumer.try_pop() {
                        assert_eq!(value, expected);
                        expected += 1;
                    } else {
                        core::hint::spin_loop();
                    }
                }
            });
            produced.join().unwrap();
            consumed.join().unwrap();
        });
    }
}
