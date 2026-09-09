# EXP-0015 — Join B observability overhead ladder

## Outcome

Three counterordered fresh-server repetitions were completed per arm. Positive latency deltas are regressions; negative throughput deltas are regressions.

**No arm's overhead is resolvable by this experiment.** After correcting for a
run-wide drift (below), every observability arm sits within the residual noise
of clean vLLM:

| Arm | Drift-corrected change vs clean | Raw paired median |
|---|---:|---:|
| scheduler | -0.07% | +0.26% |
| packed | -0.06% | +0.41% |
| sass | +0.19% | -0.87% |
| cupti | +0.51% | +0.52% |

The residual standard deviation after removing drift is **0.52%**, so nothing
in that column is distinguishable from zero. The correct claim is a bound, not
a measurement: **client-visible overhead of the full Join B stack, and of the
unfiltered CUPTI capture, is under ~0.5% on this workload — too small for this
method to resolve.**

The raw paired column is reported for continuity with the 2026-08-12 run, but
it should not be read as an arm effect; the confound below explains why its
signs disagree with the drift-corrected column, and why TTFT p99 appears to
*improve* by 6-8% in the sass and cupti arms.

## The dominant effect is execution order, not instrumentation

Output throughput declines monotonically with execution ordinal across all 15
rungs, independent of arm:

```
ordinal  1  cupti      62.05 tok/s        ordinal  9  packed     59.09
ordinal  2  sass       61.54              ordinal 10  cupti      59.11
ordinal  3  packed     61.11              ordinal 11  packed     58.84
ordinal  4  scheduler  60.30              ordinal 12  cupti      58.90
ordinal  5  clean      60.14              ordinal 13  scheduler  58.76
ordinal  6  clean      59.92              ordinal 14  clean      58.60
ordinal  7  scheduler  59.72              ordinal 15  sass       58.09
ordinal  8  sass       59.37
```

A linear fit on ordinal alone gives **-0.246 tok/s per rung, R^2 = 0.920** --
a **-6.4% decline from the first rung to the last**, roughly twelve times the
largest arm effect. Counterordering gives each arm three distinct positions
within its repeat, but the drift runs across the *whole* experiment rather than
resetting per repeat, so pairing an arm against the clean rung of the same
repeat does not remove it: in repeat 1, cupti ran first and clean ran last,
which alone accounts for cupti's apparent +3.2% advantage in that repeat.

This is why the drift-corrected column above is the one to trust, and why the
2026-08-12 four-arm run carries the same confound.

### What the drift is not, and what it might be

- **Not the engine.** Engine RSS is flat at 2.11 GiB across all 15 rungs, and
  every rung starts a fresh server against a pinned image and revision.
- **Not simply thermal.** GPU temperature rises 57C -> 69C over the first three
  rungs and then plateaus in the 64-71C band, while throughput keeps falling
  through rung 15. Correlation with throughput is only -0.571. GPU power
  (37.3-38.0 W) and utilisation (85.9-88.9%) are flat throughout.
- **Correlates best with host memory in use**, which climbs 45.4 -> 46.5 GiB
  over the run (correlation -0.826) while engine RSS does not move -- so the
  growth is host-side (page cache, container churn), not the workload.

GB10 is a unified-memory part: host and device share the same physical memory
and the same 273 GB/s of bandwidth, and batch-1 decode is bandwidth-bound, so
rising host memory pressure is a plausible mechanism. It is only plausible:
ordinal, temperature and memory all trend together over three hours and this
design cannot separate them. Establishing cause needs a dedicated experiment
(idle-hold rungs, drop_caches between arms, or a randomised block design with
ordinal modelled explicitly).

## Comparability with 2026-08-12

The clean arm reproduces the previous run almost exactly -- **59.920 tok/s
today vs 59.808 tok/s on 2026-08-12, +0.19%** -- so the two experiments share
a baseline and the older result stands beside this one rather than being
superseded by it. (An earlier single-rung pre-flight measured 62.4 tok/s and
suggested the box had gotten ~4% faster; that reading was the cold-first-rung
artifact visible at ordinal 1 above, not a change in the hardware.)

## Workload and controls

- Qwen3-14B BF16, pinned model revision and container image
- eager execution; async scheduling and prefix caching disabled
- 64 measured requests per run, exactly 128 input and 64 output tokens each
- concurrency 8, infinite offered request rate, fixed seed `20260812`
- 8 warm-up requests excluded from every measurement
- fresh server for every arm; three counterordered repeats per arm
- the cupti arm captures **unfiltered** activity (340k+ kernel rows per rung),
  matching run-live-demo-cupti.sh rather than the filtered export path
- cupti and sass are alternatives on top of packed, never combined: this stack
  has a single CUPTI subscriber slot

## Median client observations

| Arm | Output tok/s | TTFT p99 (ms) | ITL p99 (ms) | E2E p99 (ms) |
|---|---:|---:|---:|---:|
| clean | 59.920 | 520.259 | 132.878 | 8557.995 |
| scheduler | 59.717 | 520.543 | 134.154 | 8616.845 |
| packed | 59.088 | 515.872 | 134.864 | 8696.227 |
| sass | 59.373 | 490.592 | 135.385 | 8638.371 |
| cupti | 59.114 | 483.610 | 135.719 | 8688.978 |

## Paired median changes versus clean

| Arm | Output throughput | TTFT p99 | ITL p99 | E2E p99 |
|---|---:|---:|---:|---:|
| scheduler | +0.26% | +0.05% | +0.20% | -0.33% |
| packed | +0.41% | -0.67% | +1.49% | -0.48% |
| sass | -0.87% | -6.06% | +1.89% | +0.94% |
| cupti | +0.52% | -8.00% | +0.70% | -0.66% |

## Incremental paired changes

| Added boundary | Output throughput median [range] | TTFT p99 median [range] | ITL p99 median [range] | E2E p99 median [range] |
|---|---:|---:|---:|---:|
| scheduler_vs_clean | +0.26% [-0.34, +0.28] | +0.05% [-5.56, +0.71] | +0.20% [-0.14, +0.96] | -0.33% [-0.35, +0.69] |
| packed_vs_scheduler | +0.12% [-1.05, +1.35] | -0.72% [-9.16, +5.52] | +0.53% [-0.95, +11.80] | -0.13% [-1.25, +0.92] |
| sass_vs_packed | +0.48% [-1.27, +0.70] | -4.12% [-5.73, +1.94] | -0.84% [-7.60, +0.39] | -0.67% [-0.75, +4.32] |
| cupti_vs_packed | +0.12% [+0.04, +1.54] | -6.25% [-8.56, +0.55] | -2.18% [-9.80, +0.63] | -0.18% [-1.52, -0.08] |

## Interpretation boundary

- Client TTFT is first streamed-token receipt minus request send, measured outside EngineCore.
- Client ITL is the gap between consecutive streamed-token receipts.
- Engine-step durations exist only in semantic arms and use patched Python CLOCK_MONOTONIC boundaries.
- SASS events are block-entry callbacks for only `reshape_and_cache_flash_kernel`.
- Three repetitions support an engineering point estimate, not a publication-grade confidence interval.

Raw per-run values and delta ranges are preserved in `summary.json`.
