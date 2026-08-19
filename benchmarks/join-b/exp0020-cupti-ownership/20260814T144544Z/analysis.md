# EXP-0020 — CUPTI-timed Qwen3-14B request-to-cache-kernel correlation

## Outcome

The scoped CUPTI experiment passed.

Eight concurrent Qwen3-14B requests were followed from vLLM scheduler membership through authoritative post-compaction packed rows, CUPTI runtime correlation IDs, and actual GPU start/end intervals for the BF16 `reshape_and_cache_flash_kernel`.

```text
request ID
  -> EngineCore step
  -> authoritative packed row
  -> CUDA runtime activity + correlation ID
  -> matching CUPTI kernel activity
  -> actual GPU [start, end) interval
```

This run did **not** collect Compute Sanitizer SASS block-entry events. Compute Sanitizer and a direct CUPTI subscriber both require the single CUPTI subscriber slot on this stack, so the two device observers cannot run simultaneously in one process.

## Raw measurements

| Measurement | Value |
|---|---:|
| Requests | 8 |
| Forced output lengths | 8, 12, 16, 20, 24, 28, 32, 36 |
| HTTP/output-length failures | 0 |
| Semantic records | 463 |
| Engine steps | 37 |
| Prefill-only / mixed / decode-only steps | 1 / 1 / 35 |
| Maximum steps in flight | 2 |
| Semantic gaps / loss markers | 0 / 0 |
| Scheduler-order mismatch steps | 25 |
| CUPTI records at sealed shutdown | 41,581 |
| CUPTI runtime records at sealed shutdown | 20,647 |
| Inferred all-kernel activity records | 20,934 |
| Target-kernel activities | 1,480 |
| Selected decode target activities | 1,400 |
| Target activities outside selected suffix | 80 |
| Selected graph / ordinary target activities | 1,400 / 0 |
| Selected decode steps | 35 |
| Target kernels per selected step | 40, always |
| Selected scheduled request rows | 167 |
| Geometry-inferred request blocks | 6,680 |
| Geometry-inferred padding blocks | 1,160 |
| Runtime correlation misses | 0 |
| PID / geometry / per-step count mismatches | 0 / 0 / 0 |
| CUPTI buffer exhaustion / invalid / output drops | 0 / 0 / 0 |
| Maximum GPU temperature | 42 C |

CUPTI used eight fixed 1 MiB activity buffers, so the activity-buffer allocation was bounded at 8 MiB. The runtime table was bounded at 65,536 records and retained all records in the measurement.

## Timing result

For the 1,400 selected decode cache-kernel activities:

| Timing | Value |
|---|---:|
| Kernel-duration sum | 7.440316 ms |
| Busy interval union | 7.440316 ms |
| Overlap among selected target kernels | 0 |
| Per-step target-kernel sum | 0.146816–0.287136 ms |
| Individual target-kernel duration, whole trace | 3.328–18.432 µs |
| Packed-layout event to first target GPU start | 117.618–144.900 ms |
| Last target GPU end to engine-step end | 9.165–11.523 ms |
| Target runtime-submission span per step | 0.369–13.136 ms |

The duration sum equals the busy union because all selected target activities ran on one CUPTI stream and did not overlap each other.

The roughly 118–145 ms interval before the first selected cache kernel is **not proven GPU idle time**. This agent retained raw rows only for the target kernel family. CUPTI observed 20,934 total kernel activities, but non-target intervals were deliberately filtered before storage. Therefore this run times the selected cache kernels exactly but does not yet close the entire engine-step wall-time budget.

## Clock result

The agent sampled `cuptiGetTimestamp()` between two `CLOCK_MONOTONIC` reads 64 times at measurement start and end and retained the minimum-uncertainty pair at each boundary.

| Calibration | Uncertainty | CUPTI→monotonic offset |
|---|---:|---:|
| Start | 64 ns | -1,785,907,916,133,280,896 ns |
| End | 64 ns | -1,785,907,916,133,280,896 ns |
| Observed offset drift | — | 0 ns |

The correlator used affine interpolation between the two pairs, rather than assuming a permanent fixed offset. Runtime API timestamps selected the packed-layout window; matching correlation IDs then supplied actual kernel intervals in the same normalized timeline.

## What CUPTI changed relative to EXP-0019

EXP-0019 proved same-run SASS block ownership for this cache kernel, but its graph-node callbacks were host observation points and its raw device `%globaltimer` values were used only for ordering.

EXP-0020 adds:

- actual GPU kernel start and end timestamps;
- CUDA runtime-to-kernel correlation IDs;
- graph/ordinary identity from CUPTI activities;
- normalized step-relative kernel intervals;
- explicit kernel-duration sum and busy-union accounting.

EXP-0020 does not directly carry EXP-0019's SASS block events into the same run. The evidence modes are complementary:

```text
Deep ownership mode:
  packed rows -> graph nodes -> SASS blockIdx events

Timed production mode:
  packed rows -> runtime correlation ID -> actual CUPTI kernel interval
```

## Why the ownership arithmetic still matters

Each selected decode step had one active token row per request and 40 target cache kernels, one per transformer layer. The authoritative model-runner event, not scheduler list order, supplied row ownership after `InputBatch.condense()`.

Across the selected suffix:

```text
167 scheduled rows * 40 target kernels = 6,680 request-row/kernel memberships
6,680 real row blocks + 1,160 graph-padding blocks = 7,840 launch-geometry blocks
```

These are launch-geometry inferences in EXP-0020, not observed SASS block callbacks. EXP-0019 independently observed complete device block events for a closely matched workload.

## Subscriber incompatibility gate

Four small gates rejected direct same-process coexistence before the 14B run:

1. Compute Sanitizer plus Nsight/CUPTI: SASS events appeared, but Nsight reported `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`.
2. Compute Sanitizer first, activity-only CUPTI second: activity registration returned result 39 and no CUPTI activity appeared.
3. CUPTI first, Compute Sanitizer second: CUPTI captured kernels but the SASS subscriber emitted no events.
4. `sanitizerUnsubscribe()` and `cuptiFinalize()` handoff attempts did not release a reusable subscriber slot; sanitizer-owned device allocations also were not readable through ordinary `cuMemcpyDtoH`.

The direct handoff path was removed from production source. Separate observer modes are the supported design.

## Counterevidence and hardening

The first FIFO-based successful run froze its offline join at the explicit `F` command, then the process-exit finalizer captured 51 additional shutdown runtime records. This is why `join-b-cupti.log` reports 20,596 runtime records while the final sealed table and summary contain 20,647. No target kernel was added, and the sealed table/summary counts agree.

After the run, the agent was hardened to disable both CUPTI activity kinds before its forced flush. A three-kernel CUDA smoke then retained exactly three target kernels and nine in-window runtime records, with table and summary counts unchanged at process exit. That hardening is source state after EXP-0020 and is not retroactively claimed as the binary used in this run.

The measured pre-hardening source was reconstructed into `source/agent/` and rebuilt with the same CUDA 13.2.1 image. Its SHA-256 exactly matches the agent hash in `manifest.txt`, so the measured binary remains reproducible rather than merely described.

## Measured, inferred, unknown

### Measured

- Complete request outputs and authoritative packed layouts.
- Actual target-kernel start/end, stream, graph identity, geometry, and runtime correlation IDs.
- Complete CUPTI quality counters and clock calibration.
- Twenty-five compaction-reordered decode steps.
- Zero loss, correlation, PID, geometry, or per-step kernel-count errors.

### Inferred

- Per-request block membership from authoritative row ranges plus one-dimensional launch geometry.
- Total all-kernel activity count as `records_seen - runtime_records`.

### Unknown

- Full GPU busy time for each engine step, because non-target intervals were not stored.
- Same-run direct linkage between CUPTI intervals and Compute Sanitizer SASS block events.
- Multi-stream/overlapping target attribution; this selected family used one stream.
- Generalization to attention, GEMM, fused kernels, other models, or other GPUs.
- Publication-grade CUPTI overhead under randomized repeated industry workloads.

## Next gate

Run a bounded all-kernel CUPTI mode that preserves every kernel interval, computes both per-step duration sum and interval union, and groups names into semantic families. That experiment can finally explain the full engine-step timing gap instead of only the cache-kernel slice.
