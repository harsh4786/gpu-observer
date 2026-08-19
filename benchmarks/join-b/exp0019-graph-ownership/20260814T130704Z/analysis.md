# EXP-0019 — End-to-end CUDA Graph request ownership on Qwen3-14B

## Outcome

The scoped Join B experiment passed. In one Qwen3-14B run, the observer followed requests from vLLM's final packed rows into 35 CUDA Graph decode replays and then into 7,560 SASS block-entry events without drops or ownership mismatches.

```text
request ID
  -> vLLM engine step
  -> authoritative packed row after InputBatch compaction
  -> CUDA Graph executable + replay + node
  -> reshape_and_cache blockIdx.x
  -> request-owned cache-kernel work
```

This is not yet a general request-to-any-kernel mapping. It is a correct, bounded mapping for one semantically useful cache kernel under the validated single-stream topology.

## Raw measurements

| Measurement | Value |
|---|---:|
| Requests | 8 |
| Forced output lengths | 8, 12, 16, 20, 24, 28, 32, 36 |
| HTTP/output-length failures | 0 |
| Semantic records | 466 |
| Engine steps | 38 |
| Prefill-only / mixed / decode-only steps | 1 / 2 / 35 |
| Maximum steps in flight | 2 |
| Semantic gaps / loss markers | 0 / 0 |
| Scheduler-order mismatch steps | 25 |
| Graph replays / graph nodes | 36 / 1,440 |
| Nodes per replay | 40, always |
| Selected decode replays / nodes | 35 / 1,400 |
| Distinct graph executables | 4 |
| Ordinary target launches in selected suffix | 0 |
| Device events retained | 130,752 |
| Selected device events | 7,560 |
| Device drops / sequence errors | 0 / 0 |
| Request-owned / padding blocks | 6,400 / 1,160 |
| Request-count mismatches | 0 |
| Maximum GPU temperature | 42 C |

Non-streaming client completion times rose with forced output length from 1.061 s for 8 tokens to 4.659 s for 36 tokens. These are full-response times, not TTFT measurements.

## Why the arithmetic closes

Across the selected suffix, vLLM scheduled 160 real decode rows. Qwen3-14B contributes one selected cache-kernel node per transformer layer, giving 40 nodes per replay:

```text
160 scheduled rows * 40 nodes = 6,400 request-owned blocks
6,400 owned blocks + 1,160 padded blocks = 7,560 selected events
```

Every term was obtained independently from a raw stream. The correlator did not invent missing blocks.

Padding came from vLLM graph bucket sizes:

| Graph grid X | Replays | Actual active rows represented |
|---:|---:|---|
| 8 | 18 | 8, 7, 6, or 5 |
| 4 | 8 | 4 or 3 |
| 2 | 4 | 2 |
| 1 | 6 | one pre-measurement replay plus five selected one-row replays |

The graph executable changed when the graph bucket changed. Within each executable, node handles and geometry stayed stable across replay IDs.

## Compaction evidence

Step 10 still contained eight requests in rows 0 through 7. When the shortest request finished, step 11 moved the previous last request from row 7 into row 0 and retained the other survivors:

```text
step 10 rows: [A, B, C, D, E, F, G, H]
step 11 rows: [H, B, C, D, E, F, G]
```

All seven surviving scheduler positions therefore disagreed with physical packed-row order. The authoritative model-runner event recovered the correct ownership, and every request received 40 block events for its one decode row.

## Why event-tail selection was allowed

Function-scoped SASS events carry `launch_id=0`, so ordered count partitioning is only honest if no ordinary eager target launch is mixed into the selected graph suffix.

The new passive launch observer recorded all 1,400 ordinary target launches. The last occurred at monotonic timestamp `805097171735295 ns`; the first accepted suffix packed layout occurred at `805097172883676 ns`, 1,148,381 ns later. Therefore, the accepted suffix contains graph replay work only.

All selected graph nodes also reported one context/stream identity, and each node's expected block count exactly partitioned the device-event tail. This earns the single-stream ordered join for this run.

## Observer boundaries

- vLLM semantic timestamps mark CPU-side engine and packing events on `CLOCK_MONOTONIC`; they do not measure GPU duration.
- Graph-node timestamps mark Compute Sanitizer host callbacks on `CLOCK_MONOTONIC`; they do not equal device kernel start.
- Device `%globaltimer` timestamps remain raw and unnormalized; only record order and geometry were used.
- Client timings include the serving path outside EngineCore.

No cross-clock duration is claimed here.

## Counterevidence and limitation

Ordinary launch geometry predicted 125,200 blocks, and all graph callbacks predicted 7,600 more, for 132,800 whole-process callbacks. The device counter retained 130,752. The exact 2,048 difference equals the first ordinary launch's grid, consistent with that launch running before the selected module patch became active.

That missing pre-measurement launch does not enter the selected 7,560-event suffix, whose geometry is complete. Nevertheless, the observer must not claim whole-process completeness.

## Measured, inferred, unknown

### Measured

- Complete request outputs, semantic layouts, graph replay topology, ordinary launch table, device block events, and zero drop counters.
- Exact per-request block-count agreement for all 35 selected decode steps.
- Twenty-five selected steps with scheduler-order versus packed-row mismatch.

### Inferred

- The 2,048 whole-process callback deficit is likely the first pre-patch target launch because both magnitudes and ordering match exactly.

### Unknown

- Whether ordered attribution remains valid with multiple CUDA streams or overlapping graph executions.
- Actual GPU start/end times and overlap; CUPTI is required.
- How to generalize row arithmetic to attention and GEMM tiling.
- Matched randomized performance overhead of this full graph-safe telemetry combination.

## Next gate

Add CUPTI activity intervals and correlation IDs to replace ordered-count partitioning with interval-based graph-node/device attribution. This should first preserve the exact EXP-0019 ownership result, then deliberately introduce a multi-stream or overlapping workload to test where the current single-stream assumption breaks.
