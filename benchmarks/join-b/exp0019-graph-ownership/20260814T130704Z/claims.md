# EXP-0019 claims

## Accepted, scoped claim

For Qwen3-14B BF16's `reshape_and_cache_flash_kernel` on this GB10 stack, authoritative post-compaction packed rows can be joined to CUDA Graph replay nodes and complete SASS block-entry events to recover per-request cache-kernel block ownership during a single-stream decode suffix.

Evidence:

- 35 selected decode steps and 1,400 selected graph nodes
- 7,560 selected device block events
- 6,400 request-owned blocks and 1,160 padding blocks
- zero semantic, graph-table, launch-table, or device-event drops
- zero geometry or request-count mismatches
- 25 selected steps exercised scheduler-order versus packed-row reordering
- zero ordinary target launches occurred in the selected suffix

## Rejected broader claims

This run does not establish:

- attribution for arbitrary attention, GEMM, or fused kernels;
- multi-stream graph attribution;
- device kernel start/end timing;
- instruction-level request ownership;
- publication-grade instrumentation overhead;
- generalization beyond this model, kernel family, vLLM build, or GB10 stack.

## Counterevidence retained

The complete function counter observed 130,752 callbacks, while ordinary launch geometry plus graph-node geometry predicted 132,800. The 2,048-event difference equals the first ordinary target launch's grid exactly and occurs before the measured decode suffix. This is consistent with the first launch executing before the selected module patch became active; it does not affect the suffix join, but whole-process callback completeness is not claimed.
