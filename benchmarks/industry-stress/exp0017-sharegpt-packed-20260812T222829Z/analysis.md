# EXP-0017 — Production packed-row attribution under AIPerf ShareGPT

## Outcome

**Pass for scheduler-to-packed-row ownership under production execution.**

Qwen3-14B served 8 warm-up and 100 measured real ShareGPT requests with vLLM's
default asynchronous scheduler and FULL_AND_PIECEWISE CUDA Graph mode. Every
engine step emitted a complete scheduler view and a complete authoritative
post-compaction packed layout through the bounded semantic ring.

This experiment does **not** claim production device-block attribution. The
preceding EXP-0016 gates showed that the selected Compute Sanitizer SASS
block-event patch is unsafe for this cache kernel under CUDA Graph replay.

## Causal path tested

```text
AIPerf client request
  -> Python EngineCore scheduler decision
  -> SchedulerOutput with step ID and request slices
  -> GPUModelRunner persistent InputBatch update/compaction
  -> authoritative packed token-row ranges
  -> CUDA Graph model execution
```

The first four arrows were directly observed. CUDA Graph execution was verified
from vLLM startup logs, but no device block callback was active in this run.

## Configuration

- Hardware: NVIDIA GB10 / DGX Spark
- Model: `Qwen/Qwen3-14B`
- Model revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Runtime image: `nvcr.io/nvidia/vllm:26.05-py3`
- Precision: BF16
- Attention backend: FlashAttention 2
- vLLM scheduling: asynchronous, production default
- Execution: FULL_AND_PIECEWISE CUDA Graphs
- Prefix caching: disabled
- Maximum model length: 4096
- KV-cache allocation: 8 GiB
- Maximum active sequences: 32
- Semantic ABI: v2, fixed 96-byte records
- Semantic ring: 65,536 records, bounded and nonblocking

Exact versions, hashes, command configuration, and process topology are in
`manifest.txt`, `async-scheduling-evidence.txt`, and
`cuda-graph-evidence.txt`.

## Workload

AIPerf 0.10.0 replayed the frozen real ShareGPT raw-payload file used by
EXP-0012:

- 12 distinct prompts, sequentially repeated
- prompt lengths at most 90 model input tokens
- streaming OpenAI-compatible chat endpoint
- 128 output tokens per request
- 8 warm-up requests excluded by AIPerf
- 100 measured requests
- concurrency 8
- fixed output length and deterministic temperature
- payload SHA256:
  `894a9eab9526b3db4b0029029465305a96cc737e7131aefa3413de7e19cc1cca`

## Client measurements

| Metric | Result |
|---|---:|
| Measured requests | 100 |
| Errors | 0 |
| Output tokens | 12,800 |
| Benchmark duration | 205.562 s |
| Request throughput | 0.4865 requests/s |
| Output throughput | 62.268 tokens/s |
| TTFT average / p99 | 282.275 / 312.128 ms |
| ITL average / p99 | 122.329 / 122.742 ms |
| Request latency average / p99 | 15,818.060 / 15,883.121 ms |

AIPerf is the observer for these values. Its endpoints are client request send,
first streamed-token receipt, later streamed-token receipts, and response
completion. These are not EngineCore or GPU-only durations.

The earlier EXP-0012 baseline used the same request file but a maximum model
length of 8192 and was run on a different day/code state. Its 63.053 output
tokens/s is useful context only; the point difference is not a matched overhead
measurement. EXP-0015 remains the controlled observer-overhead experiment.

## Semantic and packed-layout integrity

| Measurement | Value |
|---|---:|
| Raw semantic records | 33,063 |
| Engine-step begins / ends | 1,805 / 1,805 |
| Scheduler request slices | 13,824 |
| Packed-layout headers | 1,805 |
| Packed request slices | 13,824 |
| Sequence gaps | 0 |
| Loss markers / producer drops | 0 / 0 |
| Maximum simultaneous host steps | 2 |
| Scheduled tokens | 17,289 |
| Prefill / decode tokens | 3,573 / 13,716 |
| Prefill-only / decode-only / mixed steps | 14 / 1,778 / 13 |
| Maximum request slices in a step | 8 |

The 13,716 decode-token count equals 108 total requests times 127 decode
iterations: each request's first output token is produced by its prefill, then
127 later tokens are produced by decode steps. Semantic totals include the
eight warm-up requests, while AIPerf's performance report excludes them.

The typed Rust validator additionally required:

- exactly one scheduler membership entry per packed request;
- equal phase and scheduled-token counts on both sides;
- contiguous packed row ranges starting at zero;
- exact coverage of the scheduled token count;
- strictly increasing packing generation;
- a complete step end with success status.

All checks passed.

## Production reordering counterexample

The validator found one step with four positional mismatches:

```text
step 129 scheduler order:
  754f, 197a, 01af, e534

step 129 final packed rows:
  row 0 -> e534
  row 1 -> 01af
  row 2 -> 197a
  row 3 -> 754f
```

All four requests carried one decode token. Membership, token counts, and phases
matched; only physical packed-row order changed. Therefore a scheduler-order
guess would have misattributed every row in that step, while the final
GPUModelRunner emitter recovered the correct ownership.

The raw records are preserved in `step-129-reordering-evidence.txt`; the
machine-readable mismatch summary is in `packed-order-mismatches.txt`.

## Async timing semantics

The maximum in-flight depth was two. For example, step 129 began and published
its packed layout before step 128 emitted its end event.

```text
step 128 begin -> packed layout
step 129 begin -> packed layout
step 128 end
step 129 end
```

This is host pipeline overlap, not proof that two kernels execute concurrently.
Actual device start/end timing requires CUPTI. It also explains why naive
overlapping begin/end temporal brackets cannot safely assign launches in async
mode.

## Engine-step measurements

| Class | Steps | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| All | 1,805 | 243.783 ms | 243.531 ms | 248.348 ms | 254.922 ms |
| Prefill only | 14 | 175.109 ms | 173.908 ms | 238.513 ms | 238.513 ms |
| Decode only | 1,778 | 244.130 ms | 243.538 ms | 248.269 ms | 250.066 ms |
| Mixed | 13 | 270.240 ms | 272.160 ms | 288.631 ms | 288.631 ms |

These are patched Python EngineCore intervals on CLOCK_MONOTONIC. In async mode
they include pipeline waiting and overlap and are not GPU busy time.

## System telemetry

| Measurement | Value |
|---|---:|
| Samples | 206 |
| Mean / maximum GPU utilization | 93.19% / 96% |
| Maximum GPU temperature | 63 C |
| Mean / maximum GPU power | 29.49 / 36.74 W |
| Maximum EngineCore RSS | 2,230,896 KiB |
| Maximum EngineCore threads | 125 |
| Minimum available system memory | 77,022,908 KiB |

The server was stopped after capture. The post-run GPU state was idle.

## Measured, inferred, unknown

**Measured**

- AIPerf completed 100 measured requests with zero errors and exactly 128 output
  tokens each.
- Every one of 1,805 scheduler steps had a complete authoritative packed layout.
- The semantic ring had zero gaps, drops, or loss markers.
- Async scheduling produced two host steps in flight.
- Step 129 reordered all four scheduled request positions.

**Inferred**

- The step-129 permutation is consistent with persistent InputBatch
  compaction/reuse after earlier requests complete.
- Publishing ownership after the model runner's final batch mutation is the
  correct abstraction boundary for packed-row attribution.

**Unknown**

- Which CUDA Graph kernel blocks consumed each packed row in this run.
- Actual GPU start/end time and busy time for these steps.
- Whether a graph-safe SASS callback design can preserve per-launch identity.
- Publication-grade performance overhead under production graphs; this is one
  instrumented run, not a randomized matched ladder.

## Gate disposition

```text
production async scheduler semantics             PASS
post-compaction packed-row ownership             PASS
bounded transport integrity                      PASS
scheduler-order counterexample                   PASS
production CUDA Graph execution                  PASS
packed row -> SASS block under CUDA Graphs        NOT MEASURED / EXP-0016 FAILED
packed step -> actual GPU kernel timing           NEXT: bounded CUPTI/NVTX join
```
