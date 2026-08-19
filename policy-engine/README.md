# Policy engine

This crate is `no_std` and consumes normalized compact step summaries. The
first detector recognizes one condition: an interactive decode shares a
GPU-expensive step with a large background prefill.

Detection returns a value; it does not mutate vLLM. A separate controller must
log condition, action, previous value, new value, and observed result.
