# EXP0012 — Qwen3-14B mixed-prefill interference

Date: 2026-08-11  
Machine: NVIDIA DGX Spark / GB10  
Status: completed, preliminary single-run performance evidence plus causal CUPTI capture

## Executive result

A fixed stream of 1,031–1,032-token background prompts degraded an otherwise identical
100-request interactive AIPerf run:

| Interactive metric | Arm A: no background | Arm B: fixed background | Change |
|---|---:|---:|---:|
| TTFT average | 386.638 ms | 414.767 ms | +7.28% |
| TTFT p99 | 414.401 ms | 674.224 ms | +62.70% |
| ITL average | 119.934 ms | 128.714 ms | +7.32% |
| ITL p99 | 120.366 ms | 132.688 ms | +10.24% |
| Request latency average | 15,618.243 ms | 16,761.408 ms | +7.32% |
| Request latency p99 | 15,657.516 ms | 17,226.955 ms | +10.02% |
| Interactive output throughput | 63.053 tok/s | 58.888 tok/s | -6.61% |
| Benchmark duration | 203.004 s | 217.361 s | +7.07% |

The separate causal capture explains the regression at device level:

- A normal decode step used a median 120.563 ms of GPU busy time.
- The step mixing a 1,031-token prefill with eight decode tokens used 391.628 ms,
  or 3.25 times the ordinary decode median.
- The following decode step's kernels began 366.722 ms after its scheduler begin,
  versus a 119.759 ms ordinary-decode median.
- That following decode step's semantic interval reached 490.397 ms.

The direct mechanism is therefore:

    long prefill enters a mixed engine step
        ↓
    mixed step occupies the GPU for about 271 ms more than a normal decode
        ↓
    production async scheduling has already admitted the following decode step
        ↓
    that decode waits in the GPU work queue
        ↓
    interactive token latency increases

This establishes Join A for this capture: request slices and scheduler steps are connected
to every recorded GPU kernel. It does not establish Join B: kernels in a mixed step remain
honestly many-to-many with the requests in that batch.

## Experimental question

Does a long prefill sharing vLLM execution with short decoders create measurable interactive
tail-latency degradation, and can the scheduler-to-GPU join explain the mechanism?

## Fixed system configuration

- GPU: NVIDIA GB10, compute capability 12.1
- Driver: 580.173.02
- OS: Ubuntu 24.04.4 LTS, aarch64
- Container: nvcr.io/nvidia/vllm:26.05-py3
- Container digest: sha256:654e563e727be1968487d453de0733fb074cb59adf424c779d0f9e7cfcb2b6b6
- vLLM: 0.20.1+7124b12a.nv26.5, V1 engine
- Model: Qwen/Qwen3-14B, BF16
- Model revision: 40c069824f4251a91eefaf281ebe4c544efd3e18
- max model length: 8192
- KV cache allocation: 8 GiB
- max sequences: 32
- prefix caching: disabled
- execution: production async scheduling
- compilation mode: FULL_AND_PIECEWISE CUDA Graphs
- attention backend: FlashAttention 2
- semantic transport: nonblocking fixed-record shared-memory ring
- AIPerf: 0.10.0, source commit bf70bd968f44b7d34026554e8dc629d39440d106
- Nsight Systems: 2025.3.2.474

The full machine manifest is in final-environment.txt. Exact code and binary hashes are in
final-code-sha256.txt.

## Workloads

### Interactive workload

- Source: frozen ShareGPT corpus
- Exact payload file: controlled-workload/interactive-payloads-corrected.jsonl
- 12 distinct prompts, sequentially repeated by AIPerf
- All selected prompt lengths are at most 90 model input tokens
- 128 output tokens per request
- streaming chat endpoint
- 8 warmup requests
- 100 measured requests
- concurrency 8
- fresh vLLM server for each performance arm

### Background workload

- Exact payload file: controlled-workload/background-long-payloads.jsonl
- Four real ShareGPT prompts
- Model input lengths: 1,031, 1,031, 1,032, and 1,032 tokens
- 16 output tokens per request
- 45 requests total
- one request every four seconds
- first arrival 500 ms after AIPerf reported the measured phase start
- four prompts cycled in fixed order
- all 45 responses completed and contained the streaming DONE marker
- no background curl stderr files were nonempty

The corrected interactive payload SHA256 is
894a9eab9526b3db4b0029029465305a96cc737e7131aefa3413de7e19cc1cca.
The background payload SHA256 is
3b556e56d79d69540e95e30a02a84a4197e15efb1a121c732103bcef78ace440.
The raw ShareGPT corpus SHA256 is
35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4.

## Matched performance design

Arm A and Arm B used fresh, otherwise identical servers and the exact same interactive
payload file. The only intended serving-load difference was the fixed background arrival
stream in Arm B. Both arms kept semantic tracing enabled.

| Integrity measure | Arm A | Arm B |
|---|---:|---:|
| Interactive records | 100 | 100 |
| Interactive errors | 0 | 0 |
| Output tokens | 12,800 | 12,800 |
| Semantic records | 17,460 | 18,180 |
| Semantic engine steps | 1,818 | 1,818 |
| Sequence gaps | 0 | 0 |
| Loss markers | 0 | 0 |
| Maximum semantic steps in flight | 2 | 2 |
| Peak GPU temperature | 58 C | 64 C |

The thermal abort threshold was 85 C and was never approached.

## Semantic evidence

| Step class | Arm A count | Arm A p50 | Arm B count | Arm B p50 |
|---|---:|---:|---:|---:|
| Prefill-only | 14 | 248.289 ms | 9 | 251.485 ms |
| Decode-only | 1,790 | 238.970 ms | 1,746 | 241.265 ms |
| Mixed prefill/decode | 14 | 276.298 ms | 63 | 517.208 ms |

Background traffic did not merely raise a generic utilization metric. It increased mixed-step
frequency from 14 to 63 and nearly doubled the mixed-step median semantic interval.

The semantic interval in production async mode is not pure GPU duration. It begins immediately
after the Python scheduler chooses the batch and ends when that batch's future is resolved.
With maximum in-flight depth two, intervals overlap. CUPTI is required to separate device work
from this pipeline timing.

## CUPTI causal capture

The bounded causal workload started eight short 16-token decodes, then injected one
1,031-token prompt 500 ms later. It ran under the same production async and CUDA Graph mode.

Capture integrity:

| Item | Value |
|---|---:|
| Successful responses | 9/9 |
| Semantic records | 186 |
| Semantic steps | 21 |
| Maximum steps in flight | 2 |
| CUPTI kernel activities | 11,429 |
| CUDA runtime activities | 3,006 |
| NVTX events | 395 |
| CUDA streams | 1 |
| Kernels assigned to semantic steps | 11,429/11,429 |
| Kernels without runtime correlation | 0 |
| Clock offset drift | 24 ns |
| Clock-pair uncertainty | 784/800 ns |

### Join mechanism

The production-async join does not bracket overlapping step intervals and guess.

1. The semantic emitter records the scheduler decision, step ID, request hashes, phase, and
   per-request scheduled token counts.
2. vLLM execution emits NVTX scopes for preprocess, forward, postprocess, and sample.
3. The capture observed exactly 21 semantic steps and exactly 21 forward, postprocess, and
   sample ranges, in execution order.
4. CUDA runtime calls inside each NVTX scope supply correlation IDs.
5. CUPTI kernel activities carrying those correlation IDs are assigned to that semantic step.
6. Request slices on the step create the honest many-to-many request-to-kernel relation.

The analyzer rejects missing one-to-one scopes or a kernel assigned to two steps. It does not
silently fall back to ambiguous temporal bracketing.

### Critical timeline

| Step | Composition | Semantic interval | GPU busy | GPU-start lag |
|---|---|---:|---:|---:|
| 133 | 8 decode tokens | 266.441 ms | 119.112 ms | 120.232 ms |
| 134 | 1,031 prefill + 8 decode | 512.886 ms | 391.628 ms | 118.636 ms |
| 135 | 9 decode tokens | 490.397 ms | 123.433 ms | 366.722 ms |
| 136 | 9 decode tokens | 244.209 ms | 121.002 ms | 122.931 ms |

Step 135 is the interactive victim. Its own GPU work is ordinary in size; its latency is high
because the long mixed step ahead of it holds the only observed CUDA stream.

The complete per-step request IDs and device measurements are in the successful CUPTI
attempt's cupti-step-join.tsv. The summary is in cupti-analysis.md.

## Failed attempt retained

Attempt async-mixed-20260811T155723Z failed before valid collection.

Evidence:

- VLLM_NVTX_SCOPES_FOR_PROFILING was enabled.
- The pinned NGC image's vLLM imports the separate Python nvtx module.
- That module is absent from the image.
- The first warmup returned a streamed EngineCore error.
- The semantic trace and Nsight collection were empty, and Nsight reported that collection
  stop was not allowed in that state.

The fix was a 22-line compatibility shim at vllm-adapter/nvtx.py using PyTorch's installed
torch.cuda.nvtx push/pop API. The shim was GPU-smoke-tested in the exact image before retry.
The successful attempt is async-mixed-20260811T160327Z.

This failed attempt is not included in performance or CUPTI metrics.

## What is measured, inferred, and unknown

### Measured

- Arm B's interactive TTFT p99 was 62.70% higher than Arm A.
- Arm B had 4.5 times as many mixed steps.
- The long mixed step consumed 391.628 ms of device busy time.
- The following decode began GPU execution 366.722 ms after scheduling.
- Every CUPTI kernel activity in the causal capture was correlated to a semantic step.
- Semantic transport recorded no gaps, drops, or loss flags.
- GPU temperatures remained safe.

### Inferred from measured timing

The long mixed prefill is the direct queueing cause of the following decode delay on this
single-stream capture. The extra GPU-start delay, about 247 ms over the ordinary median,
matches the extra device occupancy created by the long mixed step closely enough to support
this causal interpretation.

### Not established

- Per-block, per-warp, or per-instruction ownership by an individual request
- Join B for BF16 FlashAttention or cuBLAS kernels
- A production overhead claim for CUPTI/Nsight; the causal capture is diagnostic
- Unified-memory migration as a contributor
- Generalization to every ShareGPT distribution, model, backend, or serving policy
- Statistical confidence suitable for publication from one Arm A and one Arm B run
- Closed-loop p99 improvement; no policy action was applied in EXP0012

## Adversarial limitations

1. The performance result is one matched run per arm. One hundred requests makes p99 a fragile
   order statistic. It is demo-quality evidence, not a publishable confidence interval.
2. The interactive workload repeats 12 controlled short prompts. AIPerf is the runner, but this
   is not an official MLPerf or universal industry configuration.
3. Both performance arms include semantic tracing. A separate overhead ladder is still needed
   for clean vLLM versus semantic-only versus host probes versus CUPTI.
4. The current production-async semantic-to-NVTX mapping validates count and order rather than
   carrying the step ID inside the NVTX label. The analyzer fails closed on a mismatch, but a
   future patch should propagate step ID to the execution range.
5. CUPTI/Nsight changes execution timing. Its role here is mechanism validation, not the
   uninstrumented performance comparison.
6. Request-to-kernel attribution remains batch-level. Claiming finer precision would be wrong.

## Decision supported by EXP0012

The initial controller should target one explicit condition:

    a prefill above a configured token threshold
    + currently active interactive decodes
    + recent interactive tail latency above target

The first action should be one reversible policy, such as delaying or limiting admission of
background prefills. The next matched experiment should compare fixed scheduling against that
single policy while bounding throughput loss.

Before calling the controller result publishable, run multiple fresh-server repetitions in a
predeclared order, increase the measured interactive request count, and report confidence
intervals. The same frozen files and hashes should be reused.

## Important artifact paths

- performance/arm-a-baseline/metric-summary.json
- performance/arm-b-interference/metric-summary.json
- performance/comparison.json
- performance/arm-a-baseline/semantic-step-stats.tsv
- performance/arm-b-interference/semantic-step-stats.tsv
- cupti/latest-async-mixed.txt
- cupti/async-mixed-20260811T160327Z/cupti-step-join.tsv
- cupti/async-mixed-20260811T160327Z/cupti-analysis.md
- cupti/async-mixed-20260811T160327Z/nsys-qwen14-async-mixed.nsys-rep
- cupti/async-mixed-20260811T160327Z/nsys-qwen14-async-mixed.sqlite
- final-environment.txt
- final-code-sha256.txt
- FINAL-SHA256SUMS

Raw ShareGPT input and sparse shared-memory ring files are intentionally kept outside any
phone-synced Obsidian vault.
