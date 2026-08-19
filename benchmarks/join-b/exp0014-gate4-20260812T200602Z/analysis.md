# EXP-0014 Gate 4 — Authoritative packed-row Join B

## Question

Can a bounded shared-memory event emitted at the final `GPUModelRunner` packing
boundary replace the temporary file oracle and recover per-request ownership of
device blocks in the production BF16 `reshape_and_cache_flash_kernel`?

## Result

**Yes, for this kernel and this eager, synchronous vLLM configuration.**

The validated chain is:

```text
Python SchedulerOutput
  -> step ID and request membership

GPUModelRunner after InputBatch compaction/reordering
  -> authoritative request-to-packed-row ranges

Compute Sanitizer block-entry probe
  -> launch ID and blockIdx.x

Rust Join B validator
  -> step -> launch -> packed row -> request
```

The file oracle used in Gate 3 was removed. Both scheduler semantics and final
packed-row ownership traveled through the same bounded SPSC shared-memory ring.

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
- Target kernel: `reshape_and_cache_flash_kernel`
- Workload: eight short requests plus one delayed 1,031-token prefill
- Semantic wire ABI: v2, fixed 96-byte records

Exact commands, inputs, revisions, and hashes are in `manifest.txt` and
`checksums.sha256`.

## Measured evidence

### Semantic stream

| Measurement | Value |
|---|---:|
| Semantic records | 345 |
| Engine-step begins / ends | 19 / 19 |
| Scheduler request slices | 144 |
| Packed-layout headers | 19 |
| Authoritative packed slices | 144 |
| Complete packed steps | 19 / 19 |
| Producer drops | 0 |
| Semantic sequence gaps | 0 |
| Target launches before packed-layout publication | 0 |

The new ownership records contributed 163 fixed records, or 15,648 raw bytes,
for the measured workload. The shared ring remains fixed at startup; no
variable-length object enters its hot path.

### Device and join integrity

| Measurement | Value |
|---|---:|
| Target kernel launches | 760 |
| Launches assigned to engine steps | 760 |
| Complete measured launches | 760 |
| Device block events | 58,400 |
| Device write attempts | 58,400 |
| Device drops | 0 |
| Device sequence errors | 0 |
| Orphan events | 0 |
| Duplicate or missing blocks | 0 |
| Geometry mismatches | 0 |
| Token-grid mismatches | 0 |
| Unmapped blocks | 0 |
| Per-request event-count mismatches | 0 |

Every measured target launch produced one unique block event per packed token
row. Every event was mapped to exactly one authoritative request row range.

### Mixed steps

Two mixed prefill/decode steps were observed:

```text
step 3: 225 tokens = 224 prefill + 1 decode
step 5: 1,039 tokens = 1,031 prefill + 8 decode
```

Step 5 mapped the eight decode rows to `[0,8)` and the long prefill to
`[8,1039)`. Across 40 target launches, the long request therefore owned
`1,031 * 40 = 41,240` block-entry events.

### Reproduced scheduler-order counterexample

One of 19 steps reordered all eight active request positions.

Scheduler membership order in step 18 began with request `0x775e...` and ended
with request `0x5423...`. The final packed layout was:

```text
row 0 -> request 0x5423...
row 1 -> request 0x775e...
row 2 -> request 0x6234...
...
row 7 -> request 0x511f...
```

Join B used those final row ranges and assigned exactly 40 events to each
request, matching the 40 target launches. This is the key result: the new path
works precisely in the transition where the rejected scheduler-order method
would misattribute every row.

## Accounting caveat

The sanitizer's process-lifetime target summary reports 144,400 expected block
callbacks and 142,352 actual callbacks, a difference of exactly 2,048. Source
inspection shows that expected launch counters start before the target module is
patched; the first target launch is therefore included in expected blocks but
cannot execute the newly installed callback. The armed measurement window uses
a reset launch-ID table and event buffer and is internally exact at 58,400
expected and retained events across 760 launches.

This caveat does not invalidate Gate 4, but the process-lifetime summary should
be revised in a later tooling cleanup so pre-patch launches are labeled rather
than appearing as missing callbacks.

## Mechanism and memory discipline

1. `EngineCore.step()` publishes the scheduler step and stores its published
   step ID on that specific `SchedulerOutput` object.
2. `GPUModelRunner` updates and condenses its persistent `InputBatch`, applies
   any backend reorder, then reads the final `req_ids` order.
3. A preallocated ctypes array computes dense `[row_begin,row_end)` ranges.
4. One native call validates the complete batch, reserves header plus all slices
   atomically, writes fixed records, and publishes once.
5. If the ring is full, the whole packed layout drops visibly; inference never
   blocks and strict Join B rejects the incomplete trace.
6. Offline Rust validation requires contiguous total row coverage, unique
   scheduler membership, matching phase/token counts, increasing packing
   generations, and complete device geometry.

The Python adapter preallocates 1,024 packed entries at 40 bytes each (40,960
bytes). The ring capacity is fixed at 65,536 records rather than growing with
traffic.

## What this proves

- Scheduler order is not a safe proxy for physical packed-row order.
- Final `GPUModelRunner` order is a sufficient ownership source for this kernel.
- `blockIdx.x` is a deterministic packed-token-row coordinate for the observed
  `reshape_and_cache_flash_kernel` launches.
- The three-way join can survive a real request-completion compaction event with
  no file oracle, loss, or many-to-one ambiguity.

## What this does not prove

- It does not generalize to arbitrary GEMM, attention, or sampling kernels.
- It does not establish ownership under CUDA Graph padding or replay.
- It does not test async scheduling or cross-process worker configurations.
- It does not yet measure the adapter's end-to-end TTFT, ITL, p99, or throughput
  overhead in matched repeated runs.
- Device timestamps remain raw global-timer values here; ownership uses explicit
  launch IDs and host launch bracketing, not normalized device-time ordering.

## Gate disposition

```text
final packer order -> request row ranges       PASS
launch ID -> target kernel invocation          PASS
blockIdx.x -> packed token row                  PASS
packed row -> individual request                PASS
scheduler-order counterexample reproduced       PASS
loss/geometry/per-request integrity              PASS
general request ownership for arbitrary kernels NOT CLAIMED
```

The next experiment should isolate observer overhead with matched repetitions:
semantic step events only versus semantic plus packed-layout events, followed by
the same comparison with the selected device probe enabled.
