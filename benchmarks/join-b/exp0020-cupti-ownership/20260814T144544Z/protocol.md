# EXP-0020 protocol — CUPTI-timed request ownership

## Causal question

Can a production CUDA Graph trace recover this mapping without ordered SASS-event partitioning?

```text
vLLM request
  -> authoritative post-compaction packed row
  -> CUPTI runtime correlation ID
  -> actual target-kernel GPU interval
```

## Controlled workload

- Model: `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`
- BF16, FlashAttention 2, vLLM V1 asynchronous scheduling
- `FULL_AND_PIECEWISE` CUDA Graph mode
- Prefix caching disabled
- Eight concurrent requests with `temperature=0` and `ignore_eos=true`
- Forced output lengths: 8, 12, 16, 20, 24, 28, 32, 36 tokens
- Target: BF16 `reshape_and_cache_flash_kernel`

Different completion lengths force `InputBatch` compaction while decode remains active.

## Observers

1. Patched vLLM emits engine-step membership and authoritative post-compaction packed row ranges using `CLOCK_MONOTONIC`.
2. CUPTI Runtime Activity records API start/end, PID, TID, callback ID, return value, and correlation ID.
3. CUPTI Concurrent Kernel Activity records actual GPU start/end, stream, correlation ID, graph/node identity, launch geometry, and kernel name.
4. The agent samples CUPTI and monotonic clocks at both measurement boundaries and performs affine normalization offline.

## Bounded resources

- Eight fixed activity buffers × 1 MiB = 8 MiB.
- Runtime table capacity: 65,536 records.
- Offline input cap: 64 MiB per activity/runtime file.
- Offline record cap: 1,000,000 rows.
- Semantic ring and packed-slice counts remain bounded by their existing ABI limits.
- Loss, exhaustion, invalid-row, and correlation-miss counters are explicit hard gates.

## Measurement window

The preload library is present during process creation but allocates no trace files until the selected EngineCore receives `S` through its per-PID FIFO.

The harness performs:

1. model load, compile, graph capture, and one warmup request;
2. warmup semantic-ring drain;
3. CUPTI start in the resolved EngineCore PID;
4. eight concurrent measured requests;
5. semantic capture;
6. CUPTI disable/flush command;
7. offline correlation;
8. server shutdown and artifact sealing.

## Execution

From the repository root:

```bash
cargo build --release -p gpu-observer-collector --bin join_b_cupti_cache

docker run --rm --user 1000:1000 \
  -v "$PWD/cupti-agent/activity:/src" \
  -w /src \
  nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04 \
  make BUILD_DIR=build-cuda1321 CUDA_PATH=/usr/local/cuda all

benchmarks/run-join-b-cupti-ownership.sh
```

## Offline join

For each target activity:

1. Find the CUPTI runtime row with the same correlation ID.
2. Normalize runtime and kernel timestamps by affine interpolation between start/end clock pairs.
3. Use runtime submission time to choose the latest authoritative packed-layout window that still lies inside the engine step.
4. Attach the matching actual kernel interval to that step.
5. Validate one-dimensional row geometry and graph padding.
6. Report kernel-duration sum separately from interval-union busy time.

## Acceptance gates

- Every request returns HTTP 200 and exactly its forced output length.
- Semantic records have no sequence gaps or loss markers.
- Every step has a complete authoritative packed layout.
- At least one selected decode step differs from scheduler order.
- Each selected decode step has exactly 40 target activities.
- Every target correlation ID resolves to one runtime record.
- Runtime PID matches the semantic EngineCore PID.
- Target geometry is one-dimensional and covers all scheduled rows.
- CUPTI registration, runtime, and kernel activity enablement succeed.
- Activity buffers have no exhaustion, invalid rows, or drops.
- Clock calibration brackets every selected target timestamp.
- Request-row/kernel arithmetic closes without mismatch.

## Explicit non-gate

Compute Sanitizer SASS events are not required and cannot be collected in the same process with this direct CUPTI mode on the tested stack.
