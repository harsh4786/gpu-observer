# EXP-0022 — complete single-request lifecycle on Qwen3-14B

## Question

Can one request be joined across the actual observer boundaries from client send,
through vLLM admission and scheduling, through physical GPU execution, and back to
the client’s first streamed token?

This experiment deliberately distinguishes observation points that are often
collapsed into one vague word such as “latency”:

```text
Rust client starts socket write
  -> OpenAI chat handler begins after HTTP/JSON decode
  -> EngineCore calls Scheduler.add_request
  -> Python scheduler selects the request in an engine step
  -> first physical GPU kernel starts
  -> last physical GPU kernel for the token-producing step ends
  -> EngineCore accepts the model output token
  -> frontend yields the SSE chunk
  -> Rust client finishes reading that HTTP chunk
```

## Result

**Pass.** The accepted run preserved exact request and token identity across all
four observers. It matched 6,340 actual kernel executions to 12 engine steps with
zero semantic drops, CUPTI drops, invalid records, unmatched correlations,
unassigned kernels, or fatal CUDA errors.

The first-token lifecycle was:

| Interval | Time |
|---|---:|
| Client send begin → frontend handler entry | 0.596 ms |
| Frontend handler entry → EngineCore admission | 5.907 ms |
| EngineCore admission → first scheduler selection | 0.147 ms |
| Scheduler selection → first GPU kernel start | 0.942 ms |
| First → last GPU kernel in the token-producing step | 143.304 ms |
| Last GPU kernel → EngineCore accepts token | 1.669 ms |
| EngineCore accepts token → frontend yields SSE | 2.320 ms |
| Frontend yields SSE → client receives chunk | 0.321 ms |
| End-to-end client TTFT | **155.205 ms** |

These disjoint endpoint differences sum to the measured TTFT. The frontend
endpoint is after HTTP decoding, not raw socket acceptance. The frontend emit
endpoint is immediately before the async generator yields, not proof that the
kernel TCP stack transmitted the bytes at that instant.

## Workload and system

- NVIDIA DGX Spark / GB10, TP=1
- `Qwen/Qwen3-14B`, BF16, pinned revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`
- NVIDIA vLLM 26.05 image with the isolated Python overlays
- production asynchronous scheduling
- `FULL_AND_PIECEWISE` CUDA Graphs
- prefix caching disabled
- explicit 8 GiB KV cache
- one deterministic 94-token rendered prompt
- 12 forced output tokens, temperature zero, EOS ignored
- one warm-up request before the measurement window

The exact request and complete environment/source hashes are in `request.json`
and `manifest.txt`.

## Observer design

### EngineCore ring

One EngineCore process owns one bounded SPSC ring. It emits fixed 96-byte records
for admission, engine-step begin/end, scheduler membership, authoritative packed
layout, and accepted output tokens.

### Frontend ring

The separate API frontend owns a separate bounded SPSC ring. It emits request
handler entry, exact output token IDs immediately before SSE yield, and request
completion. Keeping this ring separate preserves the one-producer invariant.

### Client trace

The Rust HTTP client timestamps before socket write and at HTTP chunk read
completion. It uses a fixed 8,192-record array and fixed per-frame token array;
records are written to disk only after the response completes.

### CUPTI

CUPTI Runtime and Driver API records provide launch correlation IDs. Concurrent
kernel activity provides actual device start/end timestamps. Two clock pairs map
CUPTI time affinely into `CLOCK_MONOTONIC`; start/end uncertainty was 64 ns and
measured offset drift was zero.

## Exact token evidence

The request produced 12 accepted tokens. The token IDs matched exactly at all
three downstream observation points:

```text
EngineCore accepted token
  == frontend yielded token
  == Rust client received token
```

The accepted IDs were:

```text
151667, 198, 32313, 11, 773, 358, 1184, 311, 7071, 700, 1246, 311
```

Client ITL after the first token stayed near 119.25–121.02 ms. `token-timeline.tsv`
preserves every token’s step ID and the GPU, EngineCore, frontend, and client
timestamps.

## GPU inventory

| Item | Value |
|---|---:|
| CUPTI records | 19,048 |
| API records | 12,708 |
| Kernel records | 6,340 |
| CUDA Graph kernels | 5,731 |
| Ordinary kernels | 609 |
| Kernel-duration sum | 1,464.641 ms |
| Overlap-safe busy union | 1,464.582 ms |
| Kernel overlap | 0.059 ms |
| Streams observed | 1 |
| Maximum semantic steps in flight | 2 |

The all-kernel analyzer passed its accounting invariant and mapped every retained
kernel exactly once. The GPU scope still excludes memcpy, memset, unified-memory
fault/migration, and multi-GPU collectives.

## Rejected attempt retained

The first run at `../20260820T190227Z` is preserved and marked rejected.

It exposed two integration faults:

1. The offline analyzer assumed a post-warm-up capture restarted record sequence
   numbers at zero. The SPSC ring correctly keeps a process-lifetime sequence;
   the analyzer now accepts any starting sequence but rejects every gap.
2. Admission telemetry interned the suffixed EngineCore request ID before the
   next step refreshed the frontend focus cursor. A cache hit in `_intern()` then
   skipped suffix-alias reconciliation, producing zero accepted-token records.
   Existing IDs now rerun the allocation-free alias check.

The rejected run still had valid CUPTI evidence, but it did not satisfy exact
end-to-end token identity and therefore was not promoted.

## Performance-engineering constraints

- The shared ABI remains one fixed 96-byte record.
- `gpu-observer-core` remains `no_std`; its allocation test passes.
- Frontend and EngineCore use separate SPSC rings.
- Emission is nonblocking and drop-accounted.
- Frontend token scratch storage is allocated once at initialization.
- JSON remains only at the OpenAI HTTP protocol boundary, not in the telemetry
  hot path.
- The release 5,000,000-event ring benchmark measured 59.04 million events/s,
  or 16.94 ns/event, after the ABI extension.

This was a diagnostic mechanism run, not a tracing-overhead benchmark. No
performance-improvement claim is made from it.

## Claims earned

- One request can be joined from client send through frontend, EngineCore,
  scheduler, actual GPU kernels, EngineCore output acceptance, frontend SSE
  yield, and client receipt on this pinned TP=1 stack.
- Exact token identity validates the three output-side joins.
- CUPTI explains the physical GPU interval rather than treating Python step wall
  time or host launch submission time as device execution.
- The major first-token interval in this single-request run was physical GPU work,
  not EngineCore queueing.

## Claims not earned

- The handler-entry timestamp is not raw network-arrival time.
- The SSE-yield timestamp is not kernel-level socket-transmit completion.
- Kernel time is not attributed arithmetically to one request inside arbitrary
  shared GEMM or attention kernels.
- No overhead, statistical generalization, policy improvement, unified-memory,
  or multi-GPU collective claim is made.
