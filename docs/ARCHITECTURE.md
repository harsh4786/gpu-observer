# Architecture

## Hot and cold planes

The hot plane is everything that can perturb vLLM request execution. It uses
`gpu-observer-core`, which is `no_std`. Heap use is limited to explicit,
fallible startup allocation of bounded rings and exact-capacity correlation
vectors. A producer does not allocate, lock, format, log, or issue a syscall.

The cold plane owns file I/O, JSON, names, UI payloads, and experiment
metadata. It can use `std` because it runs outside the inference-critical
thread and process boundary.

## Event representation

`EventRecord` is an AoS ingestion record:

- 40-byte common header;
- 64-byte tagged payload union;
- 104 bytes total, aligned to 8 bytes;
- `Copy`, pointer-free, and fixed width.

The representation uses numeric request, service-class, metric, and symbol IDs.
Names live in a dictionary side channel. The JSON compatibility layer currently
builds dictionaries by sorting borrowed `&str` slices and binary searching
them, avoiding duplicate owned dictionary strings.

Variable-length batch composition is flattened:

```text
engine_step_begin(step=172, expected_slices=3)
step_request_slice(step=172, request=12, phase=decode, tokens=1)
step_request_slice(step=172, request=19, phase=prefill, tokens=256)
step_request_slice(step=172, request=44, phase=decode, tokens=1)
engine_step_end(step=172)
```

Semantic ABI v3 adds two more flattened record families:

- `packed_token_row`: authoritative post-compaction token row, position, token ID, request hash, phase, and packing generation;
- `accepted_output_token`: the token ID accepted by the scheduler after model execution.

Only the explicitly focused request emits token-level records. The hot path uses a preallocated ctypes array and a fixed Rust ABI; it does not allocate one object per token.

This keeps records contiguous and prevents a `Vec` allocation per engine
step. The emitter publishes such a semantic group all-or-nothing.

## Transport topology

Use one SPSC ring per producer. Do not put vLLM, host probes, and CUPTI behind a
shared MPSC queue: a contended tail cache line would couple their latency.

Each ring has:

- power-of-two capacity and mask-based indexing;
- contiguous preallocated slots;
- producer-local head and cached tail;
- consumer-local tail and cached head;
- release/acquire publication;
- producer and consumer cursors on separate 64-byte cache lines;
- batch publication with one release store.

A full ring never blocks inference. The event is dropped, the source sequence
continues, and the next successful event sets `DROPPED_BEFORE`. The sequence
gap is the authoritative loss count.

## Correlation layout

Offline/collector correlation consumes compact records, sorts them in place,
counts each record class, and reserves output vectors exactly once.

The output is type-segregated:

- dense `StepSummary[]`;
- dense `Membership[]`, sorted by step index;
- dense `KernelExecution[]`, sorted by step then GPU start time;
- compact diagnostics.

Each step stores index ranges into the membership and kernel arrays. This is
effectively CSR-style adjacency: iteration is contiguous, and a step does not
own heap objects.

Open engine steps use a small linear vector. vLLM normally has only a handful
of worker processes, so a scan over this tiny set is cheaper and more local
than a hash table. The vector starts at 16 entries and grows fallibly.

## Time and identity

Every hot header carries:

- source timestamp;
- normalized timestamp;
- source clock ID;
- PID and TID;
- monotonically increasing source sequence;
- source and event kind;
- precision/loss flags.

CUPTI API-activity records belong to the engine step open on the same PID and
TID at submission time. CUPTI kernel activity is then joined to its API record
by CUPTI's correlation ID. This means asynchronous activity can begin after
`engine_step_end` and still retain the submitting step.

An external uprobe cannot read CUPTI's private correlation ID. Its launch
record therefore carries no `HAS_CORRELATION_ID` flag and is not the
authoritative submission-to-execution join. It independently supplies function
pointer, stream, launch geometry, loss accounting, and a way to validate CUPTI
API coverage. Any uprobe-to-CUPTI association by TID, order, and time remains a
best-effort diagnostic unless another exact identifier is established.

Host launch duration is never reported as kernel duration. CUPTI supplies
actual start and duration.

Per-step GPU timing reports three different quantities:

- kernel time sum, which double-counts overlap;
- GPU busy-time union, which does not double-count overlap;
- GPU span, from first start to last end.

## Process boundaries

The intended deployment is:

- serving frontend: bounded focused query/output MessagePack datagrams;
- vLLM worker: semantic producer only;
- eBPF/bpftime attachment: host CUDA producer;
- CUPTI agent: activity producer and clock calibration;
- Rust collector: ring consumers, raw writer, normalization, correlation;
- policy engine: consumes normalized compact summaries;
- controller adapter: performs and logs one explicitly authorized action.

The collector works without the controller. Device probes are optional and
cannot be required for the basic demo.

## Narrow ABIs

The semantic transport is a versioned, fixed-width C ABI backed by the shared SPSC ring. ABI v3 validates magic, version, record size, kind, generation, and bounded field ranges before converting bytes into typed records. Older v1 and v2 traces remain readable for their supported event kinds.

The frontend side channel uses bounded Unix datagrams and a length-prefixed raw MessagePack capture file. The collector decodes it only on the cold path with explicit byte, nesting, and container-count limits. `GPU_OBSERVER_TRACE_V2` is the sealed JSON presentation bundle, never the inference transport.
