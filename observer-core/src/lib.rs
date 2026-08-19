#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod correlation;
pub mod emitter;
pub mod event;
pub mod ring;
pub mod semantic;

pub use correlation::{
    correlate, CorrelationError, Diagnostic, DiagnosticCode, KernelExecution, KernelFlags,
    Membership, StepState, StepSummary, TraceIndex,
};
pub use emitter::{EmitOutcome, EventEmitter};
pub use event::*;
pub use ring::{Consumer, Producer, RingError, SpscRing};
pub use semantic::*;
