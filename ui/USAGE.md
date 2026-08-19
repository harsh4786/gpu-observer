# Query-to-SASS visualization

The primary UI consumes `GPU_OBSERVER_TRACE_V2`, a cold JSON bundle produced
from raw binary query frames, semantic records, and CUPTI TSV activity. JSON is
never constructed in the inference worker's semantic hot path.

The primary view is an interactive causal node-edge graph:

```text
natural-language query
  -> exact rendered prompt and tokenizer tokens
  -> Python EngineCore scheduler slices
  -> GPUModelRunner authoritative packed rows
  -> CUDA runtime submission and CUPTI actual GPU intervals
  -> validated cache-kernel block ownership
  -> matched Compute Sanitizer PC samples and SASS offsets
```

Every edge is labeled measured, reconstructed, matched replay, or unavailable.
For the shipped Qwen3-14B BF16 path the UI explicitly says that PTX is
unavailable; it never fabricates a PTX layer.

Before a bundle is written, the cold exporter requires the frontend prompt token IDs
to exactly equal the focused request's authoritative packed prefill tokens. When the
non-streaming output manifest is present, its token IDs must also exactly equal
EngineCore's accepted-token sequence. A mismatch rejects the bundle instead of drawing a
plausible but false path.

## Open the schema fixture

The checked-in fixture demonstrates the interaction only. Its header says
`schema fixture — not evidence`; its numbers are not benchmark results.

Terminal 1 on Spark:

```bash
cd /home/harsh4786/gpu-observer
python3 -m http.server 8088 --directory /home/harsh4786/gpu-observer
```

If your browser is on another computer, terminal 2 on that computer:

```bash
ssh -L 8088:127.0.0.1:8088 harsh4786@<spark-host>
```

Then open:

```text
http://127.0.0.1:8088/ui/trace.html
```

A server is required because browsers block trace `fetch()` from `file://`.
The page also accepts a local TraceBundle through **Open TraceBundle v2**.

## Produce a measured timed trace

This starts and stops the Qwen3-14B server itself, launches seven deterministic
background requests plus one focused agentic-debugging request, captures two
bounded frontend MessagePack frames, semantic ABI v3, and CUPTI actual GPU
intervals, then seals the run.

```bash
cd /home/harsh4786/gpu-observer
./benchmarks/run-query-to-sass-timed.sh
```

The script prints the run directory. Open its `trace-bundle-v2.json` with the
file picker, or serve the repository root and pass a relative trace URL.

## Add the matched SASS microscope

The deep run is deliberately separate because the direct CUPTI agent and
Compute Sanitizer compete for CUPTI's subscriber slot on this stack.

```bash
cd /home/harsh4786/gpu-observer
./benchmarks/run-query-to-sass-deep.sh \
  /home/harsh4786/gpu-observer/benchmarks/query-to-sass/<timed-run>
```

The deep overlay is accepted only when both match:

- the model/image/request/configuration replay fingerprint;
- the canonical semantic step signature with timestamps and sequence numbers
  removed.

It obtains each target function's PC and size through
`sanitizerGetFunctionPcAndSize`, checks every sampled PC lies in that range,
subtracts the function base to obtain a SASS offset, and verifies the offset
exists in the selectively extracted `cuobjdump` disassembly.

## Cold exporters

```bash
target/release/export_trace_bundle \
  --query RUN/query.msgpack.frames \
  --semantic RUN/semantic.bin \
  --activities RUN/cupti-PID.activities.tsv \
  --runtime RUN/cupti-PID.runtime.tsv \
  --summary RUN/cupti-PID.summary.tsv \
  --run RUN/run.json \
  --output RUN/trace-bundle-v2.json
```

Add `--deep DEEP/deep-replay.json` only for a signature-matched replay.

