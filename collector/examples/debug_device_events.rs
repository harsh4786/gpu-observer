use std::env;
use std::path::PathBuf;

use gpu_observer_collector::join_b_trace::load_device_events;

fn main() {
    let path = PathBuf::from(env::args().nth(1).expect("usage: debug_device_events PATH"));
    match load_device_events(&path) {
        Ok((events, quality)) => {
            println!("parsed ok: {} events", events.len());
            println!(
                "quality: headers={} retained={} write_attempts={} drops={} sequence_errors={}",
                quality.headers, quality.retained, quality.write_attempts,
                quality.drops, quality.sequence_errors
            );
            if let Some(first) = events.first() {
                println!(
                    "first: launch_id={} sequence={} kernel_slot={} block={:?} kind={} sm_id={}",
                    first.launch_id, first.sequence, first.kernel_slot,
                    first.block, first.kind, first.sm_id
                );
            }
            if let Some(last) = events.last() {
                println!(
                    "last: launch_id={} sequence={} kernel_slot={} block={:?} kind={} sm_id={}",
                    last.launch_id, last.sequence, last.kernel_slot,
                    last.block, last.kind, last.sm_id
                );
            }
        }
        Err(error) => {
            println!("parse error: {error}");
        }
    }
}
