# EXP-0013 Gate 3 — Independent packed-row ownership oracle

## Question

Can request ownership inside the production BF16 `reshape_and_cache_flash_kernel`
be reconstructed from scheduler request order plus device-side `blockIdx.x`?

## Result

**The scheduler-order hypothesis was rejected.**

The device mapping itself is exact for this kernel:

```text
launch_id -> blockIdx.x -> packed token row
```

However, the scheduler's `num_scheduled_tokens` dictionary is not an authoritative
description of packed row order. vLLM can compact or reorder `InputBatch` before
building model inputs. One of 19 measured engine steps changed the request order,
causing eight request identities to be assigned to the wrong rows by the original
Join B reconstruction.

Join B remains recoverable for this kernel only if the model-input packer emits the
authoritative request-to-row ranges after all compaction/reordering.

## Observer boundaries

```text
EngineCore scheduler patch
  -> request IDs and scheduled token counts

GPUModelRunner packing oracle
  -> actual input_batch.req_ids order
  -> actual cumulative packed row ranges

Compute Sanitizer SASS probe
  -> launch_id and blockIdx.x at block entry

Offline Rust validators
  -> launch -> engine step
  -> blockIdx.x -> packed row
  -> packed row -> request
```

The packing oracle is experiment-only. It writes fixed 64-byte binary records and
is not suitable as the final hot-path transport.

## Configuration

- Hardware: NVIDIA GB10 / DGX Spark
- Model: `Qwen/Qwen3-14B`
- Revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Runtime image: `gpu-observer/vllm-sanitizer:cuda13.2`
- vLLM: NGC 26.05, V1
- Precision: BF16
- Execution: eager
- Async scheduling: disabled
- Prefix caching: disabled
- KV-cache allocation: 8 GiB
- Target: `reshape_and_cache_flash_kernel`
- Workload: eight short requests plus one delayed 1,031-token prefill

Exact hashes and commands are in `manifest.txt`.

## Measured evidence

### Stream integrity

| Measurement | Value |
|---|---:|
| Semantic records | 182 |
| Complete measured engine steps | 19 |
| Target kernel launches | 760 |
| Target launches assigned to steps | 760 |
| Retained device block events | 58,400 |
| Device write attempts | 58,400 |
| Device drops | 0 |
| Device sequence errors | 0 |
| Orphan device events | 0 |
| Incomplete launches | 0 |
| Geometry mismatches | 0 |
| Token-grid mismatches | 0 |
| Oracle observations | 21 |
| Oracle observations assigned to measured steps | 19 |
| Unassigned oracle observations | 2 (warmup and flush) |
| Oracle range mismatches | 8 |
| Oracle flag failures | 1 |

The geometry validator passed because every target launch emitted one block event
per packed token row. The independent ownership validator failed.

### Mixed prefill/decode step that matched

Step 5 scheduled 1,039 tokens:

```text
packed rows [0,8)      -> eight decode requests, one row each
packed rows [8,1039)   -> one long prefill request, 1,031 rows
target launches         -> 40
events for long request -> 1,031 * 40 = 41,240
```

Scheduler order and actual packing order matched in this step.

### Counterexample: step 18

Scheduler-derived order:

```text
row 0 -> short request 0xdf67...
...
row 7 -> long request  0xb126...
```

Actual model-runner packing order:

```text
row 0 -> long request   0xb126...
row 1 -> short request  0xdf67...
...
row 7 -> short request  0x5ce2...
```

All eight rows therefore had the wrong request identity under the scheduler-order
reconstruction, even though block coverage and event counts were perfect.

## Interpretation

### Measured

- `grid.x == scheduled_tokens` for all 760 measured target launches.
- Every retained `blockIdx.x` was unique, in range, and tied to an explicit launch ID.
- Actual model-runner packing differed from scheduler iteration order in step 18.
- The original request ownership mapping is therefore incorrect in at least one
  normal request-completion transition.

### Inferred from source and the observed permutation

Immediately before packing, vLLM calls `InputBatch.condense()`. Its implementation
fills the smallest empty request index using the last active request. At step 18 a
short request had finished, and the long request moved from the last row to row 0.
That algorithm exactly explains the measured permutation.

The runner may also call attention-backend-specific reordering after condensation,
so production attribution must observe the final packed order rather than trying
to duplicate either policy on the host.

### Still unknown

- Whether the same row mapping survives CUDA Graph padding/replay.
- Whether other kernels expose a deterministic token dimension.
- The overhead of a production-quality packed-slice emitter.
- Whether FlashAttention internals can support comparable request ownership.
- Whether asynchronous scheduling introduces an additional step-identity handoff.

## Design decision

Do not use `scheduler_output.num_scheduled_tokens.items()` order to assign GPU
blocks to requests.

Add a bounded packed-slice event at this boundary:

```text
GPUModelRunner
  after InputBatch.condense()
  after backend-specific reorder
  before _prepare_inputs()
```

Each event must contain:

```text
step identity
request hash
packed row begin
packed row end
scheduled tokens
packing generation
```

The scheduler stream remains authoritative for queue depth, phase, service class,
and admission state. The packer stream becomes authoritative for tensor row
ownership. Device events then join to the packer ranges.

## Gate disposition

```text
blockIdx.x -> packed row                 PASS
scheduler order -> packed request order  FAIL
packed row -> request via packer oracle  RECOVERABLE
overall original Join B hypothesis       REJECTED
```

The run is intentionally preserved as a failed gate. See:

- `join-b-cache.log` for the geometry-only result.
- `packing-oracle.log` for the independent rejection.
- `packing-oracle.bin` for raw fixed-record oracle data.
- `device-probe.events.bin` for raw device events.
- `semantic.bin` for raw scheduler events.
