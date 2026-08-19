# Industry stress benchmark selection — Qwen3-14B on DGX Spark

Date: 2026-08-10

## Decision

Use the same pinned AIPerf client for two complementary tests:

1. **Controlled 1k/1k saturation:** fixed 1,000-token inputs and 1,000-token
   outputs while sweeping concurrency.
2. **BurstGPT P90 fixed-schedule replay:** a five-minute production-trace
   window with mixed input/output lengths and preserved arrival times.

The comparison of interest is clean Qwen3-14B BF16 versus the intended
always-on device sensor. Full memory tracing is excluded.

## Why these two

### 1. Controlled 1k/1k saturation

NVIDIA publishes 1,000-input/1,000-output concurrency tables in its NIM LLM
benchmarking guide. Fixed lengths isolate the serving engine: the amount of
prefill and decode work is held constant while concurrency moves the system
from latency-oriented execution toward saturation.

This answers:

- where output throughput stops scaling;
- where TTFT and ITL begin rising sharply;
- whether the probe moves that saturation knee;
- how engine-step composition and GPU work change across concurrency.

Publication run: concurrency 1, 2, 4, 8, and 16; 100 measured requests at each
point; exact output length enforced with EOS ignored. A shorter 16-request
qualification run is allowed only to validate the harness and estimate runtime.
Do not report qualification percentiles as final tail-latency evidence.

### 2. BurstGPT P90 fixed-schedule replay

BurstGPT is a public trace of 5.29 million Azure OpenAI requests over 121 days,
published at KDD 2025. AIPerf has a first-class BurstGPT loader and exact
timestamp replay.

Selected source interval:

- upstream file: `data/BurstGPT_1.csv`;
- timestamp range: `[3527100, 3527400)` seconds;
- interval duration: 300 seconds;
- valid requests: 271;
- one zero-token failed row removed; no row exceeded 8,192 total tokens;
- offered request rate: 0.903 requests/s;
- input pressure: 472.863 input tokens/s;
- output pressure: 63.050 output tokens/s;
- median/p99 ISL: 391 / 1,831 tokens;
- median/p99 OSL: 54 / 448 tokens.

This is the trace's approximately 90th-percentile aligned five-minute request
count. Its offered output rate is close to the 61.629 output tokens/s observed
for clean Qwen3-14B in EXP-0008, while its prompt work adds prefill pressure.
That makes it a bounded overload/interference test rather than an arbitrary
maximum-rate flood.

This answers:

- how queue depth and tail latency evolve during realistic bursts;
- which long prefills share steps with interactive decodes;
- whether a lightweight sensor can detect the transition into overload;
- whether a later controller improves goodput and recovery time.

## Harness and source pins

- AIPerf: v0.10.0, commit
  `bf70bd968f44b7d34026554e8dc629d39440d106`.
- BurstGPT: commit
  `d895a53bb7b8ec137d0d2fe203b335835a78c10a`.
- AIPerf environment:
  `/home/harsh4786/.venvs/gpu-observer-aiperf-0.10.0`.
- Exact dependency closure: `aiperf-requirements.freeze.txt`.
- Exact filtered trace: `burstgpt-p90-5min.csv`.
- Source and filtered-trace hashes: `dataset.sha256`.

## Important naming rule

The first test borrows the **Server-scenario principle** used by MLPerf:
increase offered load and evaluate throughput subject to latency constraints.
It is not an MLPerf result. Qwen3-14B is not the pinned MLPerf model/dataset,
and we are not executing MLPerf LoadGen compliance or accuracy runs.

Use the label:

> AIPerf controlled 1k/1k saturation benchmark

Never use:

> MLPerf Qwen3-14B score

## Fixed server controls

Both tests must pin and record:

- Qwen/Qwen3-14B model revision and BF16;
- vLLM and container commit/image digest;
- CUDA, driver, kernel backend, and execution mode;
- max context, KV-cache bytes, max scheduled tokens, max sequences;
- async scheduling, prefix caching, chunked prefill, and CUDA Graph state;
- sampler settings and EOS behavior;
- CPU affinity and AIPerf client placement;
- semantic, host-probe, CUPTI, and device-probe configuration.

The first publication baseline should use production-style CUDA Graph execution.
Do not silently add `--enforce-eager` merely because existing instrumentation
is easier there.

## Instrumentation gate

Before the long runs:

1. verify one clean production-mode request;
2. verify semantic step events under the selected async/sync scheduler path;
3. verify the Compute Sanitizer block counter under CUDA Graph replay;
4. verify callback accounting and zero request failures;
5. stop if the probe changes correctness, graph capture, or output length.

If semantic events disappear under async scheduling, patch the actual async
execution boundary. Do not disable async scheduling and call the result a
production benchmark.

## Metrics

At each controlled concurrency and over the BurstGPT replay record:

- request count, failures, offered and achieved request rate;
- TTFT, ITL/TPOT, and end-to-end p50/p95/p99;
- output and total token throughput;
- SLO goodput and SLO-attainment fraction;
- queue depth, active requests, KV-cache usage;
- engine-step wall time and prefill/decode token composition;
- CUPTI overlap-safe GPU busy time;
- probe callback/event/drop counts and allocated bytes;
- CPU/GPU utilization, power, temperature, and unified-memory use;
- fixed-schedule dispatch lag for BurstGPT.

Metric definitions must come from AIPerf for both tests. Do not directly compare
AIPerf percentiles with earlier `vllm bench serve` percentiles without an
explicit definition audit.

## Primary sources

- MLCommons Inference scenarios and LoadGen:
  https://docs.mlcommons.org/inference/submission/
- MLCommons LLM server metrics:
  https://mlcommons.org/2025/04/llm-inference-v5/
- NVIDIA recommends AIPerf for generative-inference benchmarking:
  https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html
- NVIDIA sequence-length and load-control guidance:
  https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html
- NVIDIA 1k/1k concurrency results:
  https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html
- AIPerf fixed-schedule and request-rate semantics:
  https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference
- AIPerf BurstGPT replay:
  https://docs.nvidia.com/aiperf/dev/tutorials/datasets-inputs/profile-with-burst-gpt-traces
- BurstGPT upstream trace:
  https://github.com/HPMLL/BurstGPT
