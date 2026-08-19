# Steps 4 and 5: GPU timing and CUDA Graph visibility

## Bottom line

Step 4 succeeded: the trace now distinguishes CPU submission from actual GPU execution and assigns every GPU kernel to a concrete vLLM engine step.

Step 5 succeeded as an observability experiment: CUDA Graph replay compresses host launch visibility dramatically, but node-level CUPTI activity still exposes the kernels that execute. It is not yet a defensible performance benchmark or a graph-only causal comparison.

## Experimental correction

The first eager and graph captures were cold first requests. They produced 370 ms and 200 ms rounded client times, but the eager trace contained lazy library loading. Treating that difference as a graph speedup would be confirmation bias.

Those captures remain in eager and graphs as raw counterevidence. The corrected runs are warm-eager and warm-graphs:

- Prefix caching disabled.
- One identical request executed while collection was paused.
- Warm-up semantic records drained separately.
- The same request captured once.
- Synchronous scheduling retained.
- Nsight recorded only CUDA and NVTX, with CPU sampling disabled.

Both measured responses used 39 prompt tokens and 3 completion tokens and returned identical generated content.

## Warmed results

| Metric | Eager | Compiled graph mode | Change |
| --- | ---: | ---: | ---: |
| Host launch APIs inside steps | 1,146 | 134 | -88.3% |
| Device kernel records | 1,146 | 1,030 | -10.1% |
| Graph-node kernel records | 0 | 927 | n/a |
| Summed GPU kernel time | 33.684 ms | 21.529 ms | -36.1% |
| Summed overlap-safe GPU busy | 33.672 ms | 21.528 ms | -36.1% |
| Summed engine-step wall time | 37.112 ms | 25.366 ms | -31.6% |
| CUDA streams | 1 | 1 | unchanged |
| Semantic records | 9 | 9 | unchanged |
| Semantic drops | 0 | 0 | unchanged |

The one-stream traces make summed kernel time and overlap-safe busy almost equal. Both are reported because that equality will not hold once multiple streams overlap.

### Per-step GPU execution

| Mode | Step | Phase | Step wall | Kernels | Summed GPU time | Overlap-safe busy |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Eager | 4 | prefill, 39 tokens | 12.689 ms | 382 | 11.032 ms | 11.020 ms |
| Eager | 5 | decode, 1 token | 12.038 ms | 382 | 11.315 ms | 11.315 ms |
| Eager | 6 | decode, 1 token | 12.385 ms | 382 | 11.337 ms | 11.337 ms |
| Graph | 4 | prefill, 39 tokens | 8.393 ms | 326 | 6.805 ms | 6.804 ms |
| Graph | 5 | decode, 1 token | 8.533 ms | 352 | 7.341 ms | 7.341 ms |
| Graph | 6 | decode, 1 token | 8.441 ms | 352 | 7.383 ms | 7.383 ms |

All 1,146 eager kernels and all 1,030 graph kernels fell inside the corresponding semantic intervals. First device work began 0.176 to 0.466 ms after the semantic step boundary.

## What graph replay hides and what survives

Eager has a one-to-one relationship: every launch correlation maps to one kernel. The prefill step used 269 cudaLaunchKernel runtime calls, 85 cuLaunchKernelEx driver calls, and 28 cuLaunchKernel driver calls. Each decode used 381 runtime launches plus one extended driver launch.

Graph mode changes the topology:

- Prefill: 29 cudaGraphLaunch calls mapped to 253 graph-node kernels, with fanout 4 to 9, plus 73 direct launches.
- Each decode: one cudaGraphLaunch mapped to 337 device kernels, plus 15 direct launches.
- CUPTI retained exact correlation from the graph-launch runtime record to every replayed node.

Therefore a host uprobe that only counts cuLaunchKernel and cudaLaunchKernel will become badly incomplete under replay. It must trace graph launch APIs and represent one launch as a parent of many kernel activities. Device activity remains observable when node granularity is enabled.

## Clock evidence

EXP-0004 measured a 16 ns spread across 64 warmed CUPTI clock samples. For these Nsight exports, session UTC origin was converted to CLOCK_MONOTONIC using request-side realtime/monotonic pairs. Offset variation between the before and after pairs was 352 ns for eager and 496 ns for graph mode.

That is far below the 8 to 13 ms step durations. The first normalized kernel starts also land 0.4 to 0.5 ms after their semantic begin boundaries in both modes, an independent sanity check.

## The important caveat

Removing enforce-eager does two things on this vLLM build:

1. It enables CUDA Graph capture and replay.
2. It enables torch.compile and changes the operator and fusion topology.

The kernel names confirm the second effect: the graph path introduces compiled Triton fused kernels and changes GEMM variants. Thus the 36.1 percent device-time reduction is a result for the complete default compiled-graph mode, not proof that graph replay alone caused it.

A three-condition follow-up is required:

- Eager, compilation disabled, graphs disabled.
- Compiled execution with graphs disabled.
- Compiled execution with graphs enabled.

That decomposition is now the highest-value next experiment for Step 5.

## Overhead and statistical limits

Nsight warns that node-level graph tracing can cause significant overhead. This experiment deliberately uses node granularity to test observability, so its timings are instrumented timings. The overhead ladder has not been measured for these modes.

There is only one corrected measured request per mode. Rounded client time was 40 ms eager and 30 ms graph, but this is not enough for latency percentiles or a stable speedup. Do not publish the client comparison as a performance result.

## Artifact map

- warm-eager and warm-graphs: corrected measured traces, semantic binaries, clock pairs, responses, server logs, Nsight reports, and SQLite exports.
- eager and graphs: preserved cold first-request captures that exposed the protocol confound.
- warmed-per-step-results.txt: per-step wall and device calculations.
- warmed-correlation-results.txt: API-to-device fanout.
- per-step-sql-results.txt and correlation-and-kernels.txt: cold-run analysis.
- environment.txt and source-hashes.txt: reproducibility.
- workspace-tests.txt and transport-benchmark.txt: regression evidence.
- checksums.sha256: sealed artifact hashes.

## Next decision

The reliable spine is now request slice to engine step to CUPTI runtime correlation to GPU activity. Keep Join B gated. Before the 8B model, isolate compile versus graph effects and add a repeated deterministic workload so p50, p95, p99, throughput, and instrumentation overhead are measured rather than inferred.
