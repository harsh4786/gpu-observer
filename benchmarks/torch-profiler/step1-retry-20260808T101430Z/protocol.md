# Reproduction protocol

This protocol documents EXP-0001. Do not rerun it unless checking a changed
software, model, hardware, or instrumentation variable.

## Server configuration

```bash
vllm serve Qwen/Qwen3-0.6B \
  --port 8000 \
  --enforce-eager \
  --max-model-len 4096 \
  --kv-cache-memory-bytes 8G \
  --profiler-config='{"profiler":"torch","torch_profiler_dir":"/traces","torch_profiler_record_shapes":true,"torch_profiler_with_stack":false}'
```

Container image ID and model revision are pinned in `manifest.toml`.

## Request sequence

1. Confirm prefix-cache counters are zero.
2. Start the profiler with `POST /start_profile`.
3. Send `request.json` once as the cold request.
4. Snapshot prefix-cache counters.
5. Send the identical file as the warm request.
6. Snapshot counters.
7. Send the identical file as the warm repeat.
8. Snapshot counters.
9. Stop the profiler with `POST /stop_profile` and wait for trace flush.

No warmup request is sent. Each response must report 44 prompt tokens and one
completion token.

## Ground-truth extraction

GPU activities are grouped by the three
`execute_context_1(<computed_tokens>)_generation_0(0)` annotations. Count
`cat == "kernel"` events and sum their `dur` values. Compare histograms by
the complete kernel name.

## External probe validation

Resolve the EngineCore host PID and its mapped `libcuda` after every restart.
Attach the corrected Aya object to both `cuLaunchKernel` and
`cuLaunchKernelEx`. Run a unique 44-token prompt and its exact repeat. A valid
run records no malformed events and no loss markers.

## Stop conditions

Invalidate rather than combine a run if request bytes, model revision, server
flags, execution mode, cache state, profiler state, or probe coverage differs.
