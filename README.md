# GPU Observer

**See exactly what one LLM request does on the GPU** — from the prompt you type
to the tokens, scheduler steps, packed GPU rows and CUDA kernel launches it
becomes, down to the KV-cache blocks it owns. Every number on screen is labeled
by how it was obtained: *measured*, *reconstructed*, *matched*, or *unavailable*.

Built on an NVIDIA DGX Spark (GB10) with vLLM V1 and Qwen3-14B.

**Try it:** <https://harsh4786.github.io/gpu-observer/> → click
**Load sample trace**. It is a real sealed capture from the DGX Spark: one
focused request alongside 7 concurrent requests, 65 engine steps, and 2,600
measured GPU intervals of the KV-cache write kernel. No GPU needed to explore it.

## The problem

Between "request in" and "tokens out", an inference server is opaque. The
scheduler batches requests, the model runner packs and reorders their tokens
into GPU rows, and CUDA Graphs replay kernels with little host-side
visibility. Profilers show kernels, but not *whose* work they are. So basic
operator questions have no direct answer: which request is this GPU work for?
What did this prompt cost on the GPU? Why did this request slow down?

## What GPU Observer does

```text
chat prompt -> tokens -> vLLM scheduler step -> packed GPU rows
            -> CUDA kernel launches (CUPTI) -> request-owned KV-cache blocks
```

- **Live view.** Type a prompt and watch the causal graph build: the tokenized
  prompt, each engine step and its phase, the packed rows it ran on, and a
  per-layer kernel DAG that lights up from real CUPTI launches — with per-stage
  kernel-call counts split into thinking and response tokens, and a rolling
  GPU-busy window.
- **Sealed traces.** Export one request's complete causal record
  (`GPU_OBSERVER_TRACE_V2`) and explore it offline, step by step — including
  where vLLM reordered rows between the scheduler and the GPU. This is what the
  hosted demo shows.
- **Honest evidence.** The inference hot path emits fixed-size records into
  bounded lock-free rings; nothing is drawn as measured unless an observer
  emitted it, and every join states its precision limit.

## Results

**Request-to-block ownership, directly observed** ([EXP-0019](benchmarks/join-b/exp0019-graph-ownership/20260814T130704Z/claims.md)).
For `reshape_and_cache_flash_kernel` under CUDA Graph replay, authoritative
post-compaction packed rows were joined to device block-entry events to recover
per-request cache-block ownership over a single-stream decode suffix: 35 steps,
1,400 graph nodes, 7,560 device block events (6,400 request-owned, 1,160
padding), zero drops and zero geometry or request-count mismatches, with 25 of
those steps exercising scheduler-versus-packed-row reordering.

**Request rows joined to real GPU intervals** ([EXP-0020](benchmarks/join-b/exp0020-cupti-ownership/20260814T144544Z/claims.md)).
The same rows joined to CUDA correlation IDs and actual CUPTI kernel intervals
under asynchronous CUDA Graph execution: 1,400 activities, exactly 40 per step,
zero loss or correlation misses, 64 ns clock-calibration uncertainty.

**Overhead below the measurement floor** ([EXP-0015](benchmarks/join-b/exp0015-overhead-20260909T195321Z/analysis.md)).
15 fresh-server runs across 5 configurations — unmodified vLLM, scheduler
semantics, packed rows, device block events, and unfiltered CUPTI capture of
every kernel launch — in 3 counterordered repeats (Qwen3-14B, 64 requests of
128 in / 64 out, concurrency 8). After correcting for a run-wide throughput
drift, no configuration is distinguishable from unmodified vLLM: changes of
−0.07% to +0.51% against a residual noise of 0.52%. The honest claim is a
bound — client-visible overhead is too small for this method to resolve. The
same study found that execution order alone explained 92% of the throughput
variance (−6.4% from first run to last), which is documented rather than
hidden.

## Limits

- Exact per-request ownership is validated for one kernel family
  (`reshape_and_cache_flash_kernel`) and single-stream graph replay; it is not
  claimed for attention, GEMM, or fused kernels, or for multi-stream graphs.
- CUPTI and Compute Sanitizer cannot share a process on this stack, so timed
  kernels and device block events come from two explicit modes (a mirrored
  "shadow" container supplies the live block view, off by default; enable it
  with `?shadow=1`).
- The live kernel view is specific to Qwen3-14B's layer shape; other
  architectures need their kernel structure described first.
- The CUPTI agent caps its runtime-API table at 65,536 records, which bounds a
  sealed trace to roughly 90 output tokens.
- The live system is pinned to this hardware and container image; reproducing
  it elsewhere is not yet packaged. The hosted viewer runs anywhere.

## Timeline

This repository was started on 2026-08-19; the observer, probes, joins and
experiments above were built before the AI Infra Summit Hackathon. Work done
during the hackathon window (Sept 10–16) was the live graph flow from the chat
box into the tokenized prompt, diagnosing the shadow view's batched-delivery
join and making it opt-in, the public hosted viewer, and this submission.
The full history is in the commit log.

## Data path

```text
vLLM semantics ─┐
CUDA host probes ├─> one bounded SPSC ring per source
CUPTI activity ──┘                 │
                                   v
                       merge + clock normalization
                                   │
                         raw append-only storage
                                   │
                                   v
                 dense request <-> step <-> GPU index
                          │                    │
                     report / UI       no_std policy detector
```

The inference-facing path contains no JSON, strings, locks, syscalls, or
steady-state allocations. JSON exists only in the collector's import/export
boundary so traces remain easy to inspect.

## What works

- 104-byte pointer-free `EventRecord` with a fixed 64-byte payload union.
- Separate `step_request_slice` records instead of heap-backed batch vectors.
- Preallocated power-of-two SPSC rings with cache-line-separated cursors.
- Single-event and all-or-nothing batch publication.
- Nonblocking overload behavior with sequence-gap loss accounting.
- Dense, `no_std + alloc` correlation using exact-capacity vectors.
- CUDA launch to CUPTI activity matching by `(PID, stream, correlation_id)`.
- GPU kernel sum, execution span, and overlap-safe busy-time union.
- Honest many-to-many request-to-step membership.
- Allocation-free prefill-interference detector.
- Cold JSONL validation, raw-copy, and JSON report CLI.
- Opt-in focused query capture with rendered prompt, exact token IDs, and tokenizer strings.
- Semantic records carry authoritative packed token rows and accepted output tokens.
- Sealed `GPU_OBSERVER_TRACE_V2` export with measured/reconstructed/matched evidence labels.
- Progressive query-to-token-to-step-to-kernel UI with an optional replay-matched SASS microscope.

## Quick start

```bash
cargo test --workspace

cargo run -p gpu-observer-collector -- correlate \
  --input benchmarks/fixtures/mixed-trace.jsonl \
  --report /tmp/mixed-report.json \
  --raw-copy /tmp/mixed-raw.jsonl

cargo run --release -p gpu-observer-core --example ring_bench -- \
  5000000 65536
```

Open the trace viewer without a GPU (or use the hosted copy above):

```bash
python3 -m http.server 8088 --directory .
```

Then visit `http://127.0.0.1:8088/ui/trace.html?offline=1` and click
**Load sample trace**. The live system is described in
[the visualization guide](ui/USAGE.md).

The checked-in fixture demonstrates a decode request sharing an engine step
with a 256-token background prefill. Its two GPU intervals overlap: the report
correctly records 1,600 ns summed kernel time and 1,000 ns GPU busy time.

On this DGX Spark, two unpinned five-million-event release runs measured
50.96-61.33 million events/s, or 16.31-19.62 ns/event. This is a transport
microbenchmark,
not an instrumentation-overhead claim; pinned repeated measurements and
end-to-end vLLM benchmarks are required before making such a claim.

## Workspace

| Path | Responsibility | Runtime contract |
|---|---|---|
| `observer-core/` | event layout, SPSC transport, dense correlation | `no_std + alloc`; no hot-loop allocation |
| `collector/` | JSONL compatibility, raw persistence, reports | `std`; cold path |
| `host-probes/` | CUDA runtime/driver uprobes | external attachment; bounded emission |
| `vllm-adapter/` | minimal engine-step semantic emitter | no GPU instrumentation |
| `cupti-agent/` | activity records and clock calibration | narrow C/C++ boundary |
| `device-probes/` | bpftime/eGPU experiments | optional |
| `policy-engine/` | detection and control decisions | `no_std`; observation-independent |
| `workload/` | deterministic request generation | `std`; not in inference process |
| `benchmarks/` | fixtures, baselines, overhead matrices | raw results preserved |
| `ui/` | request-to-GPU causal view | consumes reports, never hot path |

See [Architecture](docs/ARCHITECTURE.md) and
[Performance and memory contract](docs/PERFORMANCE.md).

## Current boundary

The validated production path is currently kernel-family-specific:

```text
request -> EngineCore step -> authoritative packed row
        -> CUDA runtime correlation ID -> actual CUPTI kernel interval
```

For `reshape_and_cache_flash_kernel`, a separate deep mode also maps packed
rows to SASS `blockIdx.x` events under CUDA Graph replay. Compute Sanitizer and
the direct CUPTI agent cannot coexist in one process on the tested stack because
both consume CUPTI's single subscriber slot. Accordingly, timed-CUPTI and
deep-SASS evidence are explicit modes, not a fabricated same-run join.

The query-to-SASS vertical slice is implemented for the focused request and the
validated cache-kernel family. The remaining research boundary is generalizing
authoritative ownership beyond that kernel without claiming scheduler order is
packed-row order, and measuring the completed capture ladder under industry workloads.
