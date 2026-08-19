# EXP-0002 protocol

This was a boundary-validation run, not a performance comparison.

The server used Qwen3-0.6B, BF16, eager execution, an 8 GiB KV cache, and the minimal read-only vLLM overlay. Async scheduling was left at the NGC 26.05 default. The exact container configuration is in `container-inspect-start.json`.

One deterministic non-streaming request from `request.json` requested three output tokens. The Aya probe attached to the resolved EngineCore host PID and both CUDA driver launch entry points. The semantic consumer initially failed because the ring was root-owned; that failure is preserved. A root consumer then drained the ring and found zero records.

The root-cause check used three independent facts:

1. the loaded overlay hashes matched the host sources;
2. the server log explicitly reported asynchronous scheduling enabled;
3. the pinned core selects `step_with_batch_queue` instead of `step` when the batch queue is active.

No result from this run is combined with EXP-0003 because EXP-0003 changes async scheduling.
