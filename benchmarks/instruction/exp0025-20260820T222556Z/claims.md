# EXP-0025 claim boundary

## Supported

On this pinned Qwen3-14B BF16 eager run, 14,363 sampled dynamic memory-instruction events from 760 recorded `reshape_and_cache_flash_kernel` launches were joined through exact launch IDs and `blockIdx.x` to authoritative post-compaction request rows, then to five exact offsets in the shipped `sm_120` SASS. All samples were attributed across nine requests with zero device drops, sequence errors, orphan launches, PC mismatches, geometry mismatches, ownership mismatches, or padding samples.

## Not supported

- The result does not generalize block ownership to arbitrary attention or GEMM kernels.
- The samples do not provide exact request cost, dynamic instruction counts, cache misses, or elapsed time.
- The experiment does not establish acceptable overhead: 59.8 million callbacks executed even though only 14,363 events were retained.
- This exact instruction experiment was not run under CUDA Graph replay.
- No barrier samples were observed in the selected kernel; no claim is made about other kernels.
- Static SASS annotations do not by themselves provide model-operation semantics.
