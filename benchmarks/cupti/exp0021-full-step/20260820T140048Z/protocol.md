# EXP-0021 predeclared protocol

## Question
Can bounded all-kernel CUPTI activity close the complete GPU execution budget of every measured Qwen3-14B engine step under production asynchronous CUDA Graph execution?

## Observers
- Patched vLLM: scheduler begin, authoritative packed-layout timestamp, membership, and future-resolution end on CLOCK_MONOTONIC.
- CUPTI Runtime Activity: CUDA API start/end and correlation ID in CUPTI timestamp units.
- CUPTI Concurrent Kernel Activity: actual GPU start/end, stream, graph identity, symbol, and matching correlation ID.
- Clock calibration: start/end pairs affinely map CUPTI timestamps to CLOCK_MONOTONIC.

## Derived metrics
- semantic wall = step end - scheduler begin.
- schedule-to-pack = packed-layout timestamp - scheduler begin.
- first-kernel queue = first GPU start - its matching runtime API end.
- kernel sum adds durations and may double-count overlap.
- GPU busy union merges all kernel intervals without double-counting overlap.
- GPU span = final GPU end - first GPU start.
- GPU gaps = GPU span - GPU busy union; gaps are not assigned an idle cause.
- final GPU tail = semantic step end - final GPU end.

## Acceptance gate
All eight responses must match their forced output lengths. Semantic loss, CUPTI drops, buffer exhaustion, unmatched correlations, unassigned kernels, PID mismatches, invalid intervals, empty steps, kernels outside the assigned semantic step, negative first-kernel queue intervals, or record-accounting mismatches reject the run.

## Scope
Kernel timing is measured. Kernel-family labels are reconstructed from measured symbols. This diagnostic run is not an instrumentation-overhead benchmark and makes no instruction-level ownership claim.
