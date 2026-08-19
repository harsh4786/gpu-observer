# Benchmarks

`fixtures/mixed-trace.jsonl` is deterministic schema/correlation data. It is
not a measured GPU result.

Hardware experiments must preserve raw events and record commits, model
revision, CUDA/driver, Python environment, backend, eager/graph mode, vLLM
command, workload parameters, instrumentation configuration, affinity, memory

## Persistent experiment system of record

`registry.toml` is the master index. Every hardware run receives an immutable
experiment ID before requests are sent. Its host-side directory contains:

- `manifest.toml`: machine-readable configuration and headline measurements;
- `protocol.md`: exact workload and extraction procedure;
- raw traces, requests, responses, logs, and probe output;
- `claims.md`: observations, derivations, falsified claims, and open questions;
- `analysis.md`: human-readable interpretation;
- `checksums.sha256`: integrity hashes for every artifact.

Raw artifacts are never edited or replaced. Corrected interpretations append a
new note and explicitly mark the previous claim as falsified. Derived files may
be regenerated, but their inputs and extraction command must remain recorded.

Seal and verify a run from the repository root:

```bash
benchmarks/seal-run.sh benchmarks/PATH/TO/RUN
benchmarks/verify-run.sh benchmarks/PATH/TO/RUN
```

A run is not evidence until it is registered, has an environment manifest,
contains raw artifacts, and passes checksum verification. Local persistence is
not an off-device backup; publication candidates must also be copied to
independent storage without changing their checksums.
state, and results.
