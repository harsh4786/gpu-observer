use crate::event::{EventFlags, EventRecord};
use crate::ring::Producer;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmitOutcome {
    Published,
    Dropped,
}

pub struct EventEmitter<'a> {
    producer: Producer<'a, EventRecord>,
    dropped_since_publish: u64,
    total_dropped: u64,
}

impl<'a> EventEmitter<'a> {
    pub const fn new(producer: Producer<'a, EventRecord>) -> Self {
        Self {
            producer,
            dropped_since_publish: 0,
            total_dropped: 0,
        }
    }

    #[inline]
    pub fn emit(&mut self, mut event: EventRecord) -> EmitOutcome {
        if self.dropped_since_publish != 0 {
            event.header.flags |= EventFlags::DROPPED_BEFORE;
        }
        match self.producer.try_push(event) {
            Ok(()) => {
                self.dropped_since_publish = 0;
                EmitOutcome::Published
            }
            Err(_) => {
                self.record_drop(1);
                EmitOutcome::Dropped
            }
        }
    }

    pub fn emit_batch(&mut self, events: &mut [EventRecord]) -> EmitOutcome {
        if events.is_empty() {
            return EmitOutcome::Published;
        }
        if self.dropped_since_publish != 0 {
            if let Some(first) = events.first_mut() {
                first.header.flags |= EventFlags::DROPPED_BEFORE;
            }
        }
        if self.producer.try_push_slice_all(events) {
            self.dropped_since_publish = 0;
            EmitOutcome::Published
        } else {
            self.record_drop(events.len() as u64);
            EmitOutcome::Dropped
        }
    }

    #[inline]
    pub const fn total_dropped(&self) -> u64 {
        self.total_dropped
    }

    #[inline]
    pub const fn dropped_since_publish(&self) -> u64 {
        self.dropped_since_publish
    }

    #[inline]
    fn record_drop(&mut self, count: u64) {
        self.dropped_since_publish = self.dropped_since_publish.saturating_add(count);
        self.total_dropped = self.total_dropped.saturating_add(count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EngineStepEndData, EventContext, SourceKind};
    use crate::ring::SpscRing;

    fn event(sequence: u64) -> EventRecord {
        EventRecord::engine_step_end(
            EventContext {
                source_timestamp_ns: sequence,
                timestamp_ns: sequence,
                sequence,
                pid: 42,
                tid: 43,
                clock_id: 1,
                flags: EventFlags::TIMESTAMP_NORMALIZED,
                source: SourceKind::Vllm,
            },
            EngineStepEndData {
                step_id: sequence,
                status: 0,
                reserved: [0; 7],
            },
        )
    }

    #[test]
    fn overload_is_nonblocking_and_next_record_exposes_the_gap() {
        let mut ring = SpscRing::try_new(2).unwrap();
        let (producer, mut consumer) = ring.split();
        let mut emitter = EventEmitter::new(producer);
        assert_eq!(emitter.emit(event(1)), EmitOutcome::Published);
        assert_eq!(emitter.emit(event(2)), EmitOutcome::Published);
        assert_eq!(emitter.emit(event(3)), EmitOutcome::Dropped);
        assert_eq!(consumer.try_pop().unwrap().header.sequence, 1);
        assert_eq!(emitter.emit(event(4)), EmitOutcome::Published);
        assert_eq!(consumer.try_pop().unwrap().header.sequence, 2);
        let fourth = consumer.try_pop().unwrap();
        assert_eq!(fourth.header.sequence, 4);
        assert_ne!(fourth.header.flags & EventFlags::DROPPED_BEFORE, 0);
        assert_eq!(emitter.total_dropped(), 1);
    }

    #[test]
    fn semantic_batch_is_all_or_nothing() {
        let mut ring = SpscRing::try_new(4).unwrap();
        let (producer, mut consumer) = ring.split();
        let mut emitter = EventEmitter::new(producer);
        assert_eq!(
            emitter.emit_batch(&mut [event(1), event(2), event(3)]),
            EmitOutcome::Published
        );
        assert_eq!(
            emitter.emit_batch(&mut [event(4), event(5)]),
            EmitOutcome::Dropped
        );
        assert_eq!(consumer.try_pop().unwrap().header.sequence, 1);
        assert_eq!(consumer.try_pop().unwrap().header.sequence, 2);
        assert_eq!(consumer.try_pop().unwrap().header.sequence, 3);
        assert!(consumer.try_pop().is_none());
    }
}
