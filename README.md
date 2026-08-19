# gpu-observer

Programmable request-to-GPU observability for vLLM V1 on NVIDIA DGX Spark.

The repository now implements a DGX Spark vertical slice: fixed-layout bounded
transport, vLLM V1 engine-step and authoritative packed-row events, external
CUDA launch observation, graph-safe SASS block probes for one cache-kernel
family, and a bounded CUPTI agent for actual kernel intervals. Observation and
policy remain separate, and every attribution mode states its precision limit.

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
- Semantic ABI v3 records authoritative packed token rows and accepted output tokens.
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

Open the schema-only UI fixture without starting a model:

```bash
python3 -m http.server 8088 --directory /home/harsh4786/gpu-observer
```

Then visit `http://127.0.0.1:8088/ui/trace.html`. For a measured Qwen3-14B
trace, run `./benchmarks/run-query-to-sass-timed.sh`; see
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
