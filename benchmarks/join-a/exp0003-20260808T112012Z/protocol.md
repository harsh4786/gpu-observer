# EXP-0003 reproduction protocol

## Server

Run the NGC 26.05 image with the exact individual-file overlay and native bridge mounts recorded in `container-inspect-start.json`. The vLLM arguments are:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --port 8000 \
  --enforce-eager \
  --no-async-scheduling \
  --max-model-len 4096 \
  --kv-cache-memory-bytes 8G
```

Do not proceed unless the EngineCore log says asynchronous scheduling is disabled.

## Target resolution

Resolve the EngineCore host PID with NVIDIA compute-app accounting, resolve its container PID from the container process table, and read the mapped libcuda path from the container process maps. Preserve the mapping as `process-topology.txt`.

## Capture

Start the semantic consumer against the root-owned mmap ring. In parallel, attach Aya uprobes to both `cuLaunchKernel` and `cuLaunchKernelEx` for the resolved host PID. Persist every 96-byte semantic record and every 88-byte CUDA launch record.

After both collectors are active, send `request.json` exactly once. Wait for both collectors to flush and sync their output.

## Join and validation

Run:

```bash
target/release/join_a semantic.bin cuda-launches.bin 273358
```

The third argument is the recorded host PID corresponding to semantic container PID 300. Reject the trace for any malformed record, sequence gap, loss marker, PID mismatch after normalization, unmatched slice, or unassigned launch.

Host launch timestamps measure submission only. Do not report them as GPU execution duration.
