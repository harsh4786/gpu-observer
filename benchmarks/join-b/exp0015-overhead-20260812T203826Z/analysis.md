# EXP-0015 — Join B observability overhead ladder

## Outcome

Three counterordered fresh-server repetitions were completed per arm. Positive latency deltas are regressions; negative throughput deltas are regressions.

The full Join B stack had a median output-throughput change of **-0.33% versus
clean vLLM**. The selected SASS block-event sensor itself contributed a median
**-0.28% versus packed-row semantics**, with all three paired observations
negative (`-0.21%` to `-0.75%`). Steady-state ITL p99 changed by only `+0.12%`
for the full stack. TTFT p99 was `+2.02%`, but its per-stage ranges were much
wider and three repeats are insufficient to call that a stable tail estimate.

## Observation boundaries

```text
client send
  -> client receives first streamed token       TTFT
  -> client receives each later streamed token  ITL
  -> client receives completion                 E2E and throughput

patched EngineCore
  -> step begin/end on CLOCK_MONOTONIC           semantic arms only

patched GPUModelRunner
  -> final post-compaction request row ranges    packed and SASS arms

Compute Sanitizer SASS callback
  -> one event per cache-kernel block entry      SASS arm only
```

Client timing is the authoritative observer for user-visible overhead. Engine
step timing cannot be compared to clean because the clean arm intentionally has
no semantic patch. Device callbacks establish event cost and integrity, not
complete GPU busy time.

## Workload and controls

- Qwen3-14B BF16, pinned model revision and container image
- eager execution; async scheduling and prefix caching disabled
- 64 measured requests per run
- exactly 128 input and 64 output tokens per request
- concurrency 8, infinite offered request rate, fixed seed `20260812`
- 8 warm-up requests excluded from every measurement
- fresh server for every arm
- three counterordered repeats per arm

This is a deterministic synthetic-token overhead workload. It isolates observer
cost; it is not the ShareGPT mechanism workload or an industry-representative
traffic distribution.

The preceding Gate 4 correctness run used eight short real ShareGPT-style chat
requests (programming, business, and technical questions), each forced to 16
output tokens. After 500 ms, one 1,031-input-token strategy article arrived and
generated 16 tokens. The decisive mixed step colocated its 1,031 prefill tokens
with eight interactive decode tokens.

## Median client observations

| Arm | Output tok/s | TTFT p99 (ms) | ITL p99 (ms) | E2E p99 (ms) |
|---|---:|---:|---:|---:|
| clean | 59.808 | 517.202 | 133.540 | 8571.289 |
| scheduler | 59.772 | 506.154 | 133.665 | 8576.975 |
| packed | 59.780 | 511.748 | 133.704 | 8577.181 |
| sass | 59.604 | 522.403 | 133.865 | 8612.145 |

## Paired median changes versus clean

| Arm | Output throughput | TTFT p99 | ITL p99 | E2E p99 |
|---|---:|---:|---:|---:|
| scheduler | +0.22% | -2.14% | -0.14% | -0.19% |
| packed | -0.05% | -1.05% | +0.12% | +0.07% |
| sass | -0.33% | +2.02% | +0.12% | +0.44% |

## Incremental paired changes

| Added boundary | Output throughput median [range] | TTFT p99 median [range] | ITL p99 median [range] | E2E p99 median [range] |
|---|---:|---:|---:|---:|
| scheduler_vs_clean | +0.22% [-0.63, +0.39] | -2.14% [-8.02, +7.63] | -0.14% [-0.28, +0.63] | -0.19% [-0.33, +0.55] |
| packed_vs_scheduler | -0.07% [-0.44, +0.42] | +1.11% [-7.60, +7.53] | +0.19% [-0.49, +0.26] | +0.08% [-0.37, +0.40] |
| sass_vs_packed | -0.28% [-0.75, -0.21] | +3.11% [+1.27, +7.65] | +0.21% [-0.15, +0.46] | +0.37% [+0.33, +0.98] |

## Engine and telemetry evidence

| Evidence per run | Scheduler | Packed | SASS |
|---|---:|---:|---:|
| Complete engine steps | 520 | 520 | 520 |
| Scheduler request slices | 4,096 | 4,096 | 4,096 |
| Packed-layout headers | 0 | 520 | 520 |
| Packed request slices | 0 | 4,096 | 4,096 |
| Semantic drops | 0 | 0 | 0 |
| Target SASS launches | — | — | 20,800 |
| Retained SASS events | — | — | 488,960 |
| SASS drops | — | — | 0 |
| Geometry/ownership errors | — | — | 0 |

Across all three SASS repetitions this totals 62,400 target launches and
1,466,880 retained block events, with zero drops or Join B failures. Each SASS
event file was 31,293,504 bytes.

Median Python engine-step mean time was 131.672 ms for scheduler semantics,
131.663 ms for packed semantics, and 132.043 ms with SASS block events. Step
p99 varied substantially with prefill grouping, so its three-run point
differences should not be interpreted as a precise tail overhead.

## Memory and allocation budget

- Semantic SPSC ring: 65,536 × 96-byte records plus header = 6,291,712 bytes.
- Preallocated packed ctypes array: 1,024 × 40 bytes = 40,960 bytes.
- SASS event buffer: 1,048,576 × 64 bytes = 67,108,864 bytes per CUDA context.
- All are bounded startup allocations; emission does not allocate or wait.

Median observed EngineCore maximum RSS was 2,138.5 MiB clean, 2,138.7 MiB
scheduler-only, 2,145.5 MiB packed, and 2,152.6 MiB SASS. CUDA/UVM allocations
are not necessarily charged to process RSS, so configured buffer sizes—not RSS
differences—are the authoritative memory budget. Median peak GPU temperature
was 63–65 °C across arms.

## Measured, inferred, and unknown

**Measured**

- Every run completed exactly 64 requests, 8,192 input tokens, and 4,096 output
  tokens with no request failures.
- Scheduler and packed semantic overhead were inside the observed run-to-run
  noise band.
- The SASS sensor produced a consistently negative throughput delta versus the
  packed arm, with a median cost of 0.28%.
- Full-stack output-throughput cost was 0.33% by the paired median estimate.

**Inferred**

- Most stable incremental cost comes from executing and storing one SASS event
  per selected cache-kernel block, rather than from the bounded semantic ring.
- TTFT variability is dominated by per-run prefill grouping and ordinary
  scheduling variation at this sample size; the observer may contribute, but
  this experiment cannot isolate a precise p99 TTFT cost.

**Unknown**

- Publication-grade confidence intervals require more repeats and more requests.
- Production CUDA Graph/async behavior was not measured here.
- These figures do not generalize to tracing more kernels or higher-frequency
  memory/instruction callbacks.
- Industry traces with varied prompt lengths, output lengths, and arrivals may
  expose different overhead.

## Interpretation boundary

- Client TTFT is first streamed-token receipt minus request send, measured outside EngineCore.
- Client ITL is the gap between consecutive streamed-token receipts.
- Engine-step durations exist only in semantic arms and use patched Python CLOCK_MONOTONIC boundaries.
- SASS events are block-entry callbacks for only `reshape_and_cache_flash_kernel`.
- Three repetitions support an engineering point estimate, not a publication-grade confidence interval.

Raw per-run values and delta ranges are preserved in `summary.json`.
