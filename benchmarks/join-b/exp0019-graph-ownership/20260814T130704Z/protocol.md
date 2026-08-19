# EXP-0019 protocol — CUDA Graph request-to-SASS block ownership

## Causal question

Can one same-run trace recover this scoped mapping under production async CUDA Graph execution?

```text
vLLM request
  -> authoritative post-compaction packed row
  -> CUDA Graph replay and cache-kernel node
  -> SASS block-entry event
```

## Controlled workload

- Model: `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`
- BF16, FlashAttention 2, vLLM V1 async scheduling
- `FULL_AND_PIECEWISE` CUDA Graph mode
- Prefix caching disabled
- Eight concurrent requests with `temperature=0` and `ignore_eos=true`
- Forced output lengths: 8, 12, 16, 20, 24, 28, 32, 36 tokens
- Target: BF16 `reshape_and_cache_flash_kernel`

The staggered completion lengths force `InputBatch` compaction while decode remains active.

## Observers

1. Patched vLLM emits engine-step membership and authoritative post-compaction packed row ranges using `CLOCK_MONOTONIC`.
2. Compute Sanitizer records ordinary target-kernel launch-begin callbacks using `CLOCK_MONOTONIC`.
3. Compute Sanitizer records graph-node-begin callbacks with `(graph_exec, graph_launch_id, node)` using `CLOCK_MONOTONIC`.
4. The SASS patch emits one bounded 64-byte event per block leader with `blockIdx` and raw device `%globaltimer`.

Graph callbacks are host observation points, not GPU start timestamps. Device timestamps are retained raw and used only for event order in this experiment.

## Execution

From the repository root:

```bash
cargo build --release --workspace

docker run --rm \
  -v "$PWD/device-probes/compute-sanitizer:/src" \
  -w /src \
  nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04 \
  make CUDA_PATH=/usr/local/cuda BUILD_DIR=build-graph-identity

benchmarks/run-join-b-graph-ownership.sh
```

The orchestrator starts the server in the background, waits for health, drains warmup semantics, launches all requests, captures all raw streams, runs the offline correlator, stops the server, and seals the artifacts.

## Acceptance gates

- Every request returns HTTP 200 and exactly its forced output length.
- Semantic records have no sequence gaps or loss markers.
- Every measured step has a complete authoritative packed layout.
- At least one selected decode step differs from scheduler order.
- Each graph replay has exactly 40 target nodes.
- The selected suffix contains zero ordinary target launches.
- One context/stream identity covers every selected graph node.
- Every node has complete unique 1-D block geometry.
- Device and graph tables have zero drops.
- Every scheduled row maps to exactly one authoritative request range.
- Per-request observed block count equals `scheduled_rows * 40`.
