# EXP-0020 claims

## Accepted, scoped claim

For Qwen3-14B BF16's `reshape_and_cache_flash_kernel` on this GB10 stack, the observer can join authoritative post-compaction request rows to CUDA runtime correlation IDs and actual CUPTI GPU intervals under asynchronous CUDA Graph execution.

Evidence:

- 35 selected decode steps and 1,400 selected target-kernel activities;
- exactly 40 target activities per selected step;
- 25 selected steps exercised scheduler-order versus packed-row reordering;
- zero semantic loss, CUPTI drops, correlation misses, PID mismatches, geometry errors, or step-count mismatches;
- all selected target activities were CUDA Graph nodes and none were ordinary launches;
- 64 ns start/end calibration uncertainty and zero measured monotonic-offset drift;
- 7.440316 ms selected target-kernel duration sum and busy union.

## Accepted architecture boundary

Compute Sanitizer SASS observation and a direct CUPTI subscriber cannot coexist in one process on this tested CUDA 13.2 / driver 580 stack. Both consume the single CUPTI subscriber slot. Unsubscribe, finalize, and reverse-initialization gates did not produce a valid same-run handoff.

The supported architecture therefore has two explicit modes:

- deep SASS ownership mode;
- CUPTI-timed production mode.

## Rejected broader claims

This run does not establish:

- same-run CUPTI-interval-to-SASS-event correlation;
- observed device block callbacks in EXP-0020;
- full engine-step GPU busy time;
- that the 118–145 ms before the first target kernel was GPU idle;
- attribution for arbitrary attention, GEMM, or fused kernels;
- correctness under multiple overlapping target streams;
- publication-grade CUPTI overhead;
- generalization beyond this model, kernel family, vLLM build, or GB10 stack.

## Important precision rule

The 6,680 request blocks and 1,160 padding blocks in EXP-0020 are inferred from authoritative packed rows and CUPTI launch geometry. They are not direct SASS block observations. Direct block-event evidence remains in EXP-0019.
