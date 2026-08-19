use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use gpu_observer_core::SpscRing;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn steady_state_push_and_pop_allocate_nothing() {
    let mut ring = SpscRing::try_new(1 << 12).unwrap();
    let (mut producer, mut consumer) = ring.split();

    ALLOCATIONS.store(0, Ordering::SeqCst);
    for value in 0..100_000_u64 {
        producer.try_push(black_box(value)).unwrap();
        assert_eq!(consumer.try_pop(), Some(value));
    }
    let hot_loop_allocations = ALLOCATIONS.load(Ordering::SeqCst);

    assert_eq!(hot_loop_allocations, 0);
}
