# EXP-0021 - full engine-step kernel timeline for Qwen3-14B

## First-principles question

A vLLM semantic step lasted about 240 ms in earlier traces, while host launch submission occupied only a few milliseconds. The missing question was physical:

```text
scheduler/model-runner step
  -> CUDA API submission
  -> device queue/backlog
  -> actual kernels
  -> step future resolves
```

EXP-0020 timed only the cache-write family. EXP-0021 retained every kernel interval, joined it through CUPTI API correlation IDs, and computed an overlap-safe interval union.

## Observers and endpoints

| Layer | Observer | Directly measured endpoint |
|---|---|---|
| vLLM scheduling | patched EngineCore semantic emitter | step begin/end, membership, token counts, queue depth |
| packed execution ownership | patched GPUModelRunner | authoritative post-compaction row layout |
| CUDA API | CUPTI Runtime + Driver activity | API start/end, PID/TID, correlation ID |
| GPU execution | CUPTI concurrent-kernel activity | actual kernel start/end, stream, graph IDs, symbol |
| clock join | 64-sample start/end calibration | affine CUPTI-to-CLOCK_MONOTONIC mapping |
| client correctness | curl + OpenAI response usage | HTTP result and exact output-token count |

The analyzer assigns a kernel through its API correlation ID, then assigns that API call to the latest authoritative packed-layout window whose step has not ended.

## Workload

Eight fixed ShareGPT-derived requests ran concurrently against Qwen3-14B BF16. Prompt lengths totalled 294 tokens; forced outputs were 8, 12, 16, 20, 24, 28, 32, and 36 tokens, totalling 176. Temperature was zero, EOS was ignored, prefix caching was disabled, async scheduling remained enabled, and vLLM used `FULL_AND_PIECEWISE` CUDA Graphs.

Startup, model load, compilation, graph capture, and one warmup request occurred before deferred CUPTI collection.

## Acceptance gates

The run was accepted only if:

1. all eight responses returned HTTP 200 with exact forced output lengths;
2. semantic sequence gaps and explicit loss markers were zero;
3. every step had an authoritative packed layout;
4. CUPTI registration and Runtime, Driver, and kernel enablement succeeded;
5. buffer exhaustion, invalid records, and output drops were zero;
6. every kernel had an API correlation and semantic step;
7. PID, interval, and step-boundary checks passed;
8. CUPTI record accounting exactly matched stored tables;
9. no fatal CUDA error appeared.

All gates passed.

## Raw inventory

| Measurement | Value |
|---|---:|
| Requests / prompt tokens / output tokens | 8 / 294 / 176 |
| Semantic records / steps | 466 / 38 |
| Prefill / mixed / decode steps | 1 / 2 / 35 |
| Maximum steps in flight | 2 |
| Scheduler-order mismatch steps | 26 |
| CUPTI total / API / kernel records | 55,861 / 34,470 / 21,391 |
| Graph / ordinary kernels | 19,836 / 1,555 |
| Semantic loss / CUPTI drops / buffer exhaustion | 0 / 0 / 0 |
| Unmatched API correlations / unassigned kernels | 0 / 0 |
| Kernel-duration sum | 4,645.352623 ms |
| Overlap-safe kernel busy union | 4,643.792277 ms |
| Kernel overlap | 1.560346 ms |
| Start/end clock uncertainty | 64 ns / 48 ns |
| Offset drift over capture | -8 ns |
| Maximum sampled GPU temperature | 44 C |

CUPTI used eight preallocated 1 MiB buffers. The API table was hard-capped at 65,536 records and retained 34,470.

## Decode-step result

Nearest-rank descriptive quantiles over 35 decode steps:

| Endpoint | p50 | p95 | p99 |
|---|---:|---:|---:|
| Semantic step wall | 241.152 ms | 246.269 ms | 253.555 ms |
| First GPU start minus matching API end | 119.501 ms | 121.288 ms | 130.371 ms |
| Kernel-duration sum | 120.830 ms | 122.319 ms | 122.564 ms |
| Overlap-safe kernel busy | 120.785 ms | 122.274 ms | 122.518 ms |
| Kernel span | 120.843 ms | 122.333 ms | 122.569 ms |
| Gaps inside kernel span | 0.0535 ms | 0.0857 ms | 1.1614 ms |
| Final GPU end to semantic step end | 0.1677 ms | 0.2478 ms | 2.3543 ms |

The mean decode step was 241.452 ms wall and 120.603 ms kernel-busy. This is not "50% GPU utilization." Two semantic steps were in flight, so their wall intervals overlap while the one observed CUDA stream remained almost continuously occupied within each assigned kernel span.

### What the previous anomaly was

For a typical decode iteration:

```text
Python emits/updates packed step
  -> CUDA APIs are issued
  -> that step's first kernel waits behind submitted GPU work (~119.5 ms median)
  -> its kernels occupy the stream for ~120.8 ms median
  -> the EngineCore future resolves shortly after the last kernel
```

Aya previously saw host submission, while the semantic clock covered the future's full lifetime. CUPTI supplies the actual middle interval. The delay is consistent with asynchronous double-buffering/backlog, not a mysterious unobserved 120 ms kernel.

Exact queue/submission timing remains unknown because this agent did not enable CUPTI kernel latency timestamps. The current metric is deliberately named `first_gpu_start_minus_api_end_ns`.

## Kernel-family reconstruction

| Broad family | Kernels | Duration sum | Share |
|---|---:|---:|---:|
| GEMM/projection | 6,118 | 4,591.596 ms | 98.8428% |
| Attention | 2,920 | 19.241 ms | 0.4142% |
| Layer norm | 3,078 | 9.352 ms | 0.2013% |
| KV cache write | 1,520 | 8.365 ms | 0.1801% |
| Activation | 1,520 | 6.370 ms | 0.1371% |
| Copy-like kernels | 2,792 | 5.212 ms | 0.1122% |
| Rotary + Q/K norm + sampling + other | 3,443 | 5.217 ms | 0.1123% |

Labels come from conservative symbol patterns. They aid navigation but are not compiler-proven model-operation boundaries.

## Why the first attempt was rejected

The Runtime-only attempt captured 21,351 kernels but failed to correlate 792 ordinary kernels. Unmatched symbols were direct-launch families such as Triton, CUTLASS, NVJet, and `_compute_slot_mapping_kernel`. Graph nodes correlated successfully.

This rejected:

```text
all vLLM kernels -> CUDA Runtime API
```

The corrected observer enables both Runtime and Driver activity. The accepted rerun reached zero unmatched correlations without a time-only fallback.

## Performance-engineering boundary

The live semantic data plane remains the `no_std` `gpu-observer-core` fixed-record ring; its steady-state allocation test passed with zero allocations. A release-mode 5,000,000-event local ring run measured 49.74 million events/s (20.10 ns/event).

`full_step_cupti.rs` is a cold offline CLI. It uses `std` for file I/O, while input sizes and record counts are bounded and malformed traces fail closed. Moving filesystem code to `no_std` would not reduce inference latency.

The CUPTI completion callback is not production-grade: it uses preallocated buffers but formats every row through locked stdio. NVIDIA recommends returning quickly from completed-buffer callbacks. Production should push fixed binary records into a bounded SPSC ring and let a writer thread perform disk I/O and symbol interning.

## Measured, reconstructed, unknown

### Measured

- semantic step and packed-layout timestamps;
- API/kernel records with shared correlation IDs;
- actual kernel intervals, streams, and graph identity;
- loss/drop/accounting counters;
- response lengths and client completion times.

### Reconstructed

- step ownership from API time inside authoritative packed windows;
- broad logical family from kernel symbols;
- interpretation that the approximately one-step delay is async backlog.

### Unknown

- CUPTI queued/submitted timestamps;
- device memcpy/memset/UVM intervals;
- CPU subphase decomposition;
- arbitrary per-request GEMM/attention cost;
- instruction-level behavior in this run;
- instrumentation overhead and statistical generalization.

## Next evidence gate

1. Replace TSV formatting in the CUPTI callback with a bounded binary SPSC ring.
2. Enable CUPTI kernel latency timestamps before CUDA initialization in a diagnostic mode and store queued/submitted/start/end.
3. Add memcpy and memset activities; evaluate GB10 unified-memory counters separately.
4. Repeat the short workload and validate against a matched Nsight Systems run.
5. Then run a counterbalanced, repeated AIPerf/ShareGPT clean-versus-semantic-versus-CUPTI overhead study.

## Primary method references

- [NVIDIA CUPTI Usage](https://docs.nvidia.com/cupti/main/main.html): correlation IDs, Runtime/Driver activity, asynchronous buffering, callback guidance, and flushing.
- [NVIDIA CUpti_ActivityAPI](https://docs.nvidia.com/cupti/api/structCUpti__ActivityAPI.html): API timestamps and correlation IDs.
- [NVIDIA CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html): concurrent kernels and optional queued/submitted latency timestamps.
- [MLPerf Inference submission guide](https://docs.mlcommons.org/inference/submission/): LoadGen benchmark discipline intended for the later overhead/SLO study, not claimed here.
