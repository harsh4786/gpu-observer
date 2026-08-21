# EXP-0021 claims

## Outcome

**Scoped pass:** the observer closed the all-kernel portion of the Qwen3-14B engine-step timeline under asynchronous CUDA Graph execution. It did not close memory-copy, memset, unified-memory, CPU, or instruction-level activity.

## Accepted claims

For this pinned Qwen3-14B BF16 / vLLM / GB10 run:

- all 21,391 recorded kernel executions were joined to one of 38 engine steps;
- all kernel records had a matching CUDA Runtime or Driver API correlation ID;
- every step had authoritative post-compaction packed-layout metadata;
- semantic and CUPTI transports reported zero loss, drops, buffer exhaustion, invalid records, PID mismatches, or unassigned kernels;
- CUPTI retained actual GPU intervals for 19,836 CUDA Graph kernels and 1,555 ordinary kernels;
- across 35 decode steps, median engine-step wall time was 241.152 ms, median overlap-safe kernel busy time was 120.785 ms, and median first-GPU-start minus API-return was 119.501 ms;
- median gap inside the decode kernel span was only 53.537 us;
- two engine steps were simultaneously in flight, so the approximately 119.5 ms pre-execution interval is consistent with one step of GPU backlog under asynchronous scheduling;
- 26 of 38 steps had scheduler-order versus packed-row reordering, so request ownership must use the model runner's authoritative packed layout;
- broad symbol-pattern reconstruction assigned 98.8428% of summed kernel duration to GEMM/projection kernels.

Kernel intervals, counts, clocks, and layouts are direct measurements. Async-backlog interpretation and semantic-family labels are reconstructed.

## Counterevidence that changed the implementation

The first preserved attempt was rejected:

- 21,351 kernels were captured with zero CUPTI drops;
- 792 ordinary Triton, CUTLASS, NVJet, and slot-mapping kernels lacked a Runtime API correlation record;
- all 19,796 graph kernels and 763 other ordinary kernels did match;
- the missing kernels were direct Driver API launches, not lost GPU work.

Enabling both `CUPTI_ACTIVITY_KIND_RUNTIME` and `CUPTI_ACTIVITY_KIND_DRIVER` closed all 21,391 correlations in the accepted rerun. The rejected evidence is preserved at `20260820T140048Z`.

## Claims explicitly rejected

This experiment does **not** establish:

- a complete GPU activity budget: memcpy, memset, unified-memory migration/page-fault, and collective activity were not enabled;
- exact device queue latency: `GPU start - API return` is an endpoint difference, not CUPTI's queued/submitted timestamp;
- per-request cost attribution inside arbitrary GEMM or attention kernels;
- same-run SASS/instruction evidence;
- a causal cost share for each request in a mixed kernel;
- tracing overhead, because there was no randomized clean-versus-instrumented control;
- production-safe event transport: the CUPTI completion callback still formats TSV rows;
- generalization to other models, GPUs, vLLM revisions, multi-stream workloads, or tensor-parallel collectives.

## Precision rules

- Kernel-duration sum and overlap-safe busy union are stored separately.
- Their 1.560346 ms difference is overlap, not extra elapsed time.
- Family-duration sums are symbol-based reconstructions, not logical operation boundaries.
- Queue depth zero means no requests waited in the vLLM scheduler; it does not mean the GPU command stream had no backlog.
- The accepted API table has a legacy `.runtime.tsv` suffix but contains both Runtime and Driver records.
