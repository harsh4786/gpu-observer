# Compute Sanitizer device probe

This component uses NVIDIA's public Compute Sanitizer API to subscribe to CUDA
launch/resource events and, in patching modes, insert callbacks into already
compiled SASS. It does not require PTX.

Modes, selected by `GPU_OBSERVER_SAN_MODE`:

- `subscriber`: launch/resource callbacks only; no device code is patched.
- `block_noop`: block-entry patch that returns immediately.
- `block_counter`: one device counter increment by the block leader.
- `block_event`: one bounded 64-byte event emitted by the block leader.
- `sampled_memory_barrier`: exact callback counters plus sampled bounded events.
- `full_memory`: every global/shared/local access attempts a bounded event;
  diagnostic stress only.

The GPU callback path has no allocation and never waits for output space.
Overflow increments explicit global and per-kernel drop counters. A `SIGUSR2`
sets a flush flag; the next CUDA launch copies a snapshot to `*.summary.tsv`
and `*.events.bin`. The binary stores the raw device `%globaltimer` value and
does not pretend it is host `CLOCK_MONOTONIC`.

Build against the same CUDA version as the target image:

```bash
docker run --rm \
  -v "$PWD/device-probes/compute-sanitizer:/src" \
  -w /src \
  nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04 \
  make CUDA_PATH=/usr/local/cuda
```

The standalone compatibility gate may use `GPU_OBSERVER_SAN_FLUSH_ON_SYNC=1`.
Production vLLM measurements flush only after the measured workload.

