# EXP-0018 — Graph-safe SASS block attribution on Qwen3-14B

## Outcome

**Compatibility gate passed for the selected cache kernel under production CUDA Graph replay.**

The earlier failure was not caused by the event writer or an inherently invalid
SASS patch. It was caused by treating a replayed graph node as a fresh ordinary
kernel launch and rebinding its device callback userdata through
`sanitizerSetLaunchCallbackData`.

The repaired mechanism uses two identities:

```text
stable function-level device callback pointer
  +
stable graph executable/node identity
  +
fresh graph replay launchId
```

Qwen3-14B returned eight tokens with HTTP 200, no CUDA fatal error, 114,112
selected-kernel block events, and zero event or graph-record drops.

## Terminology

CUDA Graph kernels are not newly allocated at every replay. During capture and
instantiation, CUDA constructs a graph executable containing prepared kernel
nodes, arguments, dependencies, and resources. `cudaGraphLaunch` then replays
that executable.

In this report:

- `graphExec` identifies the instantiated graph executable;
- `node` identifies a stable prepared kernel node within it;
- `graph launchId` identifies one replay epoch of that graph executable;
- `gridId` identifies the kernel grid within that replay.

The measured identity key is `(graphExec, graph launchId, node)`.

## Causal failure and repair

### Rejected path

```text
graph replay
  -> ordinary launch callback
  -> bind per-launch userdata pointer
  -> patched block callback dereferences pointer
  -> CUDA misaligned-address fault
```

A no-op callback passed because it never dereferenced userdata. Adding only a
pointer dereference and one device atomic reproduced the fault, proving that the
64-byte event writer was not required for failure.

### Passing path

```text
first function discovery
  -> allocate one stable callback state at a stable device address
  -> sanitizerSetCallbackData(function, stable_pointer)
  -> capture/instantiate graph with that stable function binding

for each replay
  -> sanitizer graph NODE_LAUNCH_BEGIN callback
  -> copy bounded host metadata:
       graphExec, launchId, node, gridId, geometry
  -> patched block callback emits through the unchanged stable pointer
  -> offline join by replay/node identity and device event order
```

No callback pointer is rewritten between replays.

## Controlled matrix

| Arm | Callback-data scope | Raw outcome |
|---|---|---|
| Block no-op | Per launch | Request passed; no userdata dereference |
| Block counter | Per launch | Failed with CUDA misaligned-address fault |
| Block counter | Function | Request passed; counter active |
| Block event | Function | Request passed; 114,112 events, zero drops |
| Block event + graph-node records | Function | Request passed; graph replay identity captured |

The first no-op and function-counter runs were mislabeled by an early harness
`fatal_count` normalization bug. Their raw HTTP, token, fatal-scan, and probe
artifacts establish the outcomes above. The interrupted `092408Z` run is not
evidence.

Relevant arms:

- `20260813T091617Z-block_noop-launch`
- `20260813T092017Z-block_counter-launch`
- `20260813T131337Z-block_counter-function`
- `20260813T131747Z-block_event-function`
- `20260813T133629Z-block_event-function`

## Final graph-identity run

Configuration:

- Hardware: NVIDIA GB10 / DGX Spark, `sm_121`
- Model: `Qwen/Qwen3-14B`
- Revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Precision: BF16
- Attention backend: FlashAttention 2
- Scheduling: vLLM V1 asynchronous scheduling
- Execution: FULL_AND_PIECEWISE CUDA Graphs
- Target: `reshape_and_cache_flash_kernel`
- Probe: one block-leader event at kernel block entry
- Device event capacity: 1,048,576 records / 64 MiB
- Graph-node table: 65,536 records / 6 MiB
- Prefix caching: disabled
- KV-cache allocation: 8 GiB

Client acceptance:

| Check | Result |
|---|---:|
| Warm-up HTTP | 200 |
| Measured HTTP | 200 |
| Completion tokens | 8 |
| Fatal CUDA matches | 0 |
| Probe flushed | Yes |
| Post-run GPU state | 38 C, 0% utilization |

Probe integrity:

| Measurement | Result |
|---|---:|
| Target launches observed | 1,320 |
| Host-counted target blocks | 115,880 |
| Actual block callbacks | 114,112 |
| Emitted device events | 114,112 |
| Device event drops | 0 |
| Callback-data failures | 0 |
| Setup-missed launches | 0 |
| Graph nodes seen / retained | 280 / 280 |
| Graph-node table drops | 0 |

The 1,768-block difference between all host-observed target geometry and actual
callbacks is consistent with the first target launch occurring before its module
was patched at launch end. This is an inference, not yet a directly labeled
pre-patch counter; the accounting should expose pre-patch blocks explicitly in
a follow-up.

## Replay identity evidence

The graph callback recorded:

| Measurement | Result |
|---|---:|
| Graph executables | 1 |
| Stable target nodes | 40 |
| Replay launch IDs | 7 (`1` through `7`) |
| Target nodes per replay | 40 |
| Duplicate nodes within a replay | 0 |
| Node-order mismatches versus replay 1 | 0 |
| Node stream values | One: default stream |
| API stream values | One: default stream |
| Grid geometry values | One: `grid=(1,1,1)`, `block=(512,1,1)` |

Eight output tokens require seven decode replays because the prefill produces
the first output token; each later token requires a decode iteration. Qwen3-14B
has 40 transformer layers, and the selected cache kernel appears once per layer,
so each decode replay produces 40 selected nodes.

The final 280 device events independently had:

- contiguous sequences `113832..114111`;
- kernel slot `4979` for every event;
- block coordinate `(0,0,0)` for every event;
- event kind `block` for every event;
- zero raw `%globaltimer` regressions;
- exactly seven consecutive groups of 40 events.

Each replay group spanned about 106.85–108.58 ms between the first layer's cache
block entry and the last layer's cache block entry. This is **not cache-kernel
duration**: it contains the intervening work of the transformer layers. The gap
from the last cache block entry of one decode replay to the first cache block
entry of the next was about 11.38–11.42 ms.

## Observer boundaries

- Device patch observes block entry and raw GPU `%globaltimer`; it does not see
  request IDs or an ordinary per-launch identity in function scope.
- Sanitizer graph callback observes host node-launch-begin on
  `CLOCK_MONOTONIC`; it provides graph/node/replay identity but not actual GPU
  kernel start or end.
- HTTP observes request success and client-visible completion; it does not
  establish GPU attribution.

The two timestamp domains were not subtracted from each other in this
experiment.

## Attribution claim and limit

For this exact run, an ordered join is justified:

```text
7 ordered graph replays
  x 40 stable one-block nodes
  = 280 graph-node records
  = final 280 complete ordered device events
```

This earns selected-kernel `device block -> graph node -> replay epoch`
attribution for the measured single-stream graph.

It does **not** yet earn full request attribution because this compatibility
harness did not capture authoritative post-compaction packed request rows. It
also does not generalize to arbitrary multi-stream graphs: concurrent node
execution can reorder device atomics relative to host node callbacks.

The general production path is:

```text
authoritative packed request rows
  -> selected cache-kernel block coordinates
  -> raw device event timestamp
  -> CUPTI actual kernel interval
  -> graphExec + replay launchId + node
```

CUPTI is the required discriminator when multiple graph streams or overlapping
kernels make order alone ambiguous.

## Measured, inferred, unknown

**Measured**

- Per-launch stateful userdata binding faults under graph replay on this stack.
- Stable function-level userdata passes for both counter and full block-event
  callbacks.
- One graph executable reused the same 40 node handles across seven replay IDs.
- The final 280 device events match the graph callback's complete one-block
  geometry and replay grouping with zero loss.

**Inferred**

- The earlier misaligned-address fault was a per-launch callback-data lifetime
  or replay-binding problem, not event-buffer alignment.
- The 1,768 missing callbacks are the initial pre-patch launch geometry.

**Unknown**

- Exact device-event to graph-node attribution under multi-stream replay without
  CUPTI intervals.
- End-to-end graph-mode request ownership until packed-row semantic capture is
  enabled in the same run.
- Publication-grade latency/throughput overhead of graph-node callbacks and
  SASS block events; this was a correctness gate, not a randomized benchmark.

## Gate disposition

```text
Qwen3-14B production CUDA Graph execution        PASS
stateful selected SASS callback                  PASS
stable function-level callback userdata          PASS
graph executable/node/replay identity             PASS
single-stream replay -> device-event join         PASS
bounded buffers and visible drops                 PASS
request row -> replayed device block              NEXT
multi-stream/general graph attribution            NEXT: CUPTI interval join
matched graph-mode overhead ladder                NOT MEASURED
```

## Raw artifacts

Primary sealed run:

`benchmarks/join-b/exp0018-graph-callback-matrix/20260813T133629Z-block_event-function`

Important files:

- `outcome.env`
- `manifest.txt`
- `device-probe.summary.tsv`
- `device-probe.graph-nodes.tsv`
- `device-probe.events.bin`
- `server-final.log`
- `checksums.sha256`

The graph callback ABI follows NVIDIA's Compute Sanitizer API graph-domain
records: <https://docs.nvidia.com/compute-sanitizer/SanitizerApiGuide/index.html>.
