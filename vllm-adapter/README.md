# vLLM adapter

The adapter exposes semantic events at a few V1 scheduler, model-input packing, and
output-acceptance boundaries. It never calls a GPU instrumentation runtime.

A step is emitted as fixed begin, request-slice, authoritative packed-token-row, accepted
output-token, and end records. Variable-length data is flattened into fixed records and
bounded groups publish all-or-nothing through the Rust SPSC ring. The EngineCore and
GPUModelRunner path performs no JSON serialization.

Focused query context is captured separately in the OpenAI frontend because that process
owns the natural-language request, rendered prompt, tokenizer, and final response. The
`gpu_observer_query_capture.py` sidecar sends at most one bounded nonblocking MessagePack
datagram for the query and one for the non-streaming output to an external Rust receiver.
It is opt-in through `GPU_OBSERVER_FOCUS_EXTERNAL_ID`; a missing, full, or oversized socket
drops telemetry instead of delaying serving.

The two identities are explicit:

- external OpenAI request ID, used by the frontend query side channel;
- internal `chatcmpl-...` request ID, hashed into fixed semantic records.

The current exact path uses synchronous vLLM scheduling. `GPUModelRunner` emits packed rows
after its persistent `InputBatch` has applied insertion, removal, and `condense()` moves.
This is authoritative GPU input order; Python scheduler order is not used as a row oracle.

The patch is isolated in overlay files so the pinned NGC Python tree can be mounted while
the image prebuilt CUDA extensions remain untouched.
