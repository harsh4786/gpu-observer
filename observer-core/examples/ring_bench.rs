use std::hint::black_box;
use std::time::Instant;

use gpu_observer_core::SpscRing;

fn main() {
    let event_count = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("event count must be an integer"))
        .unwrap_or(10_000_000_u64);
    let capacity = std::env::args()
        .nth(2)
        .map(|value| value.parse().expect("capacity must be an integer"))
        .unwrap_or(1_usize << 16);

    let mut ring = SpscRing::try_new(capacity).expect("invalid ring capacity");
    let (mut producer, mut consumer) = ring.split();
    let started = Instant::now();

    std::thread::scope(|scope| {
        scope.spawn(move || {
            for sequence in 0..event_count {
                let mut pending = sequence;
                loop {
                    match producer.try_push(pending) {
                        Ok(()) => break,
                        Err(value) => {
                            pending = value;
                            std::hint::spin_loop();
                        }
                    }
                }
            }
        });
        scope.spawn(move || {
            let mut received = 0;
            while received < event_count {
                if let Some(sequence) = consumer.try_pop() {
                    black_box(sequence);
                    received += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });
    });

    let elapsed = started.elapsed();
    let events_per_second = event_count as f64 / elapsed.as_secs_f64();
    println!(
        "events={} capacity={} elapsed_ms={:.3} events_per_second={:.0} ns_per_event={:.2}",
        event_count,
        capacity,
        elapsed.as_secs_f64() * 1_000.0,
        events_per_second,
        elapsed.as_nanos() as f64 / event_count as f64,
    );
}
