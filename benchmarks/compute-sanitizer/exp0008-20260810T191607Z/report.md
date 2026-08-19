# EXP-0008 report: Compute Sanitizer on Qwen3-14B BF16 / GB10

## Result

The central compatibility question is answered positively: NVIDIA's public
Compute Sanitizer API successfully rewrote and executed the shipped SASS-only
BF16 kernels used by Qwen3-14B on GB10 (`sm_121`). This is a real escape hatch
around bpftime's PTX requirement. It does **not** make bpftime work on those
kernels; it proves that a separate NVIDIA SASS patcher can provide device-side
programmability.

All seven gates completed without a failed inference request, semantic event
loss, callback-data failure, module-patch failure, hang, or persistent server.
The final GPU state was idle and cool.

## Observer boundaries

- Client TTFT, ITL, end-to-end latency, and throughput: the external HTTP
  client's clock.
- Engine-step wall time: the existing vLLM Python semantic patch at
  `EngineCore.step()` begin/end, using `time.monotonic_ns()`.
- CUDA launches and grid dimensions: Compute Sanitizer host launch callbacks.
- Device callback/event counts: instructions inserted into Qwen's GPU SASS.
- Device event timestamps: raw `%globaltimer` plus SM ID. They are preserved but
  are **not** normalized to host monotonic time in this experiment.
- CPU, GPU, system memory, temperature, and power: one-second host samples.

The first six rungs used the same semantic timing layer, 64 requests, 128 input
tokens, 64 output tokens, concurrency 8, seed 20260809, eager execution, and an
8 GiB KV cache. Thus `clean` means no Compute Sanitizer subscriber or SASS
patch, not absolutely uninstrumented vLLM.

## Performance evidence

| Rung | Output tok/s | Delta vs clean | TTFT p99 | ITL p99 | E2E p99 | Mean step | Step delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 61.629 | baseline | 520.127 ms | 128.941 ms | 8,317.529 ms | 127.706 ms | baseline |
| Subscriber, no patch | 61.496 | -0.215% | 519.275 ms | 130.050 ms | 8,340.252 ms | 127.963 ms | +0.201% |
| Block no-op | 61.494 | -0.219% | 477.427 ms | 130.117 ms | 8,333.980 ms | 127.990 ms | +0.222% |
| Block counter | 61.118 | -0.829% | 510.171 ms | 130.490 ms | 8,384.255 ms | 128.773 ms | +0.835% |
| Bounded block event | 60.843 | -1.275% | 496.915 ms | 131.754 ms | 8,424.228 ms | 129.348 ms | +1.285% |
| Sampled memory/barrier | 59.835 | -2.911% | 616.827 ms | 132.024 ms | 8,568.836 ms | 131.536 ms | +2.998% |

Every rung completed 64/64 requests. The non-monotonic TTFT values are not an
instrumentation speedup: this is one run per rung, and TTFT is sensitive to
which request wave contains prefill/mixed steps. In particular, block no-op and
block-event TTFT appear lower while their ITL, E2E, and engine-step costs do
not. Repetition and randomized order are required before claiming a latency
effect. The throughput and mean-step ladder is more internally coherent, but is
still exploratory rather than a confidence-bounded benchmark.

Host mean utilization stayed near 6.3%, EngineCore near 94-95% of one CPU core,
GPU utilization near 86%, and mean used system memory near 46.3 GiB for all
six comparable rungs. Maximum temperature ranged from 52-60 C; sequential
thermal history prevents interpreting those maxima as a causal instrumentation
effect.

Exact figures, including p50/p95/p99 for all three client latency families and
system metrics, are in `aggregate.tsv` and each rung's raw `client.json`,
`engine-step-stats.tsv`, and `system.csv`.

## Device evidence

| Rung | Scope | Expected blocks | Actual device callbacks | Retained | Dropped |
|---|---|---:|---:|---:|---:|
| Subscriber | no patch | 344,907,690 | 0 | 0 | 0 |
| Block no-op | all 38 observed modules | 344,611,230 | not counted by design | 0 | 0 |
| Block counter | all 38 observed modules | 344,907,690 | 344,907,690 | 0 | 0 |
| Block event | all 41 observed modules | 344,644,170 | 344,644,170 | 65,536 | 344,578,634 |
| Sampled memory/barrier | `act_and_mul` module | n/a | 2,973,086,208 | 65,536 | 576,028 sampled events |
| Full memory stress | `act_and_mul` module | n/a | 573,785,088 | 65,536 | 573,719,552 |

The strongest correctness invariant is rung 4: the sum of launched grid blocks
equals the exact device callback count, with zero difference. Rung 5 also
satisfies `actual callbacks = retained + dropped` exactly. This proves that the
block callback is block-scoped on this stack, not one invocation per thread.

Rung 6 targeted the production BF16 gated-MLP/SILU `act_and_mul_kernel` family
identified in rung 2. One module was patched. Its two matching variants produced
about 2.97 billion exact memory/barrier callbacks. Nominal 1/4096 sampling
attempted 641,564 events; the 65,536-event buffer retained its fixed capacity
and exposed the remaining 576,028 as drops.

Rung 7 was deliberately different and is not in the performance comparison: no
warmup, one request, 128 input tokens, 16 output tokens, concurrency 1. Its
request-level p50/p95/p99 all collapse to n=1. It completed in 2.360 seconds,
reported 573.8 million full memory callbacks, and demonstrated bounded failure
instead of a hang or unbounded allocation.

## Memory behavior

The event record is one 64-byte cache line. Capacity was fixed at 65,536 records
(4 MiB). The experiment version allocated that event buffer plus a fixed
262,144-byte per-kernel state table and a 40-byte header for every patched mode:
4,456,488 device bytes per CUDA context. Subscriber mode allocated no device
buffer.

This exposed one implementation defect: block no-op and block-counter did not
need the 4 MiB event allocation. It did not materially change the roughly
46.3-GiB process/system operating point, but it violates the intended narrow
allocation policy. The exact experiment source has been snapshotted under
`source/`; the working implementation should allocate event storage only for
event-producing modes.

The buffer never blocks. Once full, events are dropped and counted. The
experiment implementation also performs a global drop atomic and a per-kernel
drop atomic on every overflow; this makes the full-buffer path more expensive
than necessary. Global drops can be derived from the reservation counter,
removing one hot atomic.

## What is measured, inferred, and still unknown

Measured:

- CUDA 13.2.1's sanitizer patcher accepted `sm_121` patch cubins.
- SASS patching worked across the production Qwen BF16 execution path.
- all-module block instrumentation patched every observed module without error;
- exact callback/event/drop counts and the latency/throughput figures above;
- no semantic ring drops, request failures, or callback setup failures.

Inferred:

- the roughly monotonic throughput/mean-step costs from counter to event to
  sampled memory are consistent with increasing device callback work;
- the large sampled-mode TTFT increase likely involves prefill/mixed steps, but
  one run is insufficient to attribute it causally.

Unknown:

- confidence intervals and run-to-run variance;
- event-to-engine-step clock calibration for raw `%globaltimer` values;
- the lowest-overhead useful sampling policy when exact per-access callback
  counts are not required;
- CUDA Graph compatibility for this custom patcher;
- whether the same behavior holds for other BF16 fused libraries and future
  CUDA/driver releases.

## Important accounting limitation

For rungs 2-6, sanitizer counters cover the EngineCore process lifetime through
the final snapshot, including startup and warmup launches. Client and semantic
performance metrics cover only the measured workload because warmup semantic
records were drained. Therefore callback totals prove execution and integrity,
but are not yet a measurement-window event rate. A reset/snapshot protocol is
required before correlating exact callback totals with individual measured
steps.

## Next decisive experiments

1. Repeat clean, subscriber, block no-op, counter, event, and sampled modes at
   least three times in randomized order with a thermal stabilization rule.
2. Add a post-warmup counter snapshot/reset so device totals cover precisely the
   same request interval as client and semantic metrics.
3. Rerun sampled `act_and_mul` at 1/65536. The observed callback volume predicts
   roughly 45k samples, which should fit without drops and separate callback
   overhead from overflow accounting.
4. Normalize raw `%globaltimer` events and join them to engine steps; preserve
   the raw clock and calibration uncertainty.
5. Compare eager with CUDA Graph replay only after the reset and clock gates are
   correct.

## Artifact map

- `aggregate.tsv`: consolidated exact figures.
- `<rung>/client.json`: raw detailed request and token intervals.
- `<rung>/semantic.bin`: fixed-width raw engine-step records.
- `<rung>/device-probe.events.bin`: bounded raw device events.
- `<rung>/device-probe.summary.tsv`: per-kernel launches, callbacks, events,
  and drops.
- `<rung>/system.csv`: one-second raw utilization and memory samples.
- `<rung>/checksums.sha256`: sealed per-rung integrity manifest.
- `source/` and `source.sha256`: exact experiment implementation snapshot.

