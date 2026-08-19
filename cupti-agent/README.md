# CUPTI agent

The activity agent is the narrow C++ boundary for actual GPU timing. It emits:

- CUPTI Runtime Activity with PID, TID, API interval, return value, and correlation ID;
- Concurrent Kernel Activity with actual GPU interval, stream, graph/node identity,
  launch geometry, kernel name, and matching correlation ID;
- start/end calibration pairs for both `CLOCK_MONOTONIC` and
  `CLOCK_MONOTONIC_RAW`.

It is designed for bounded, deferred attachment:

- eight fixed 1 MiB CUPTI buffers (8 MiB total);
- at most 65,536 runtime rows;
- lazy trace-file creation only after the selected process starts collection;
- visible exhaustion, invalid-row, output-drop, and runtime-drop counters;
- disable-then-flush behavior so process shutdown cannot extend a sealed window.

## Build

Build against the exact CUDA headers used by vLLM:

```bash
docker run --rm --user 1000:1000 \
  -v "$PWD/cupti-agent/activity:/src" \
  -w /src \
  nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04 \
  make BUILD_DIR=build-cuda1321 CUDA_PATH=/usr/local/cuda all
```

## Control

With `GPU_OBSERVER_CUPTI_DEFER=1`, every preloaded process creates only a
temporary per-PID FIFO. It does not touch the 8 MiB activity buffers or create
trace files until it receives `S`:

```bash
printf S > /tmp/gpu-observer-cupti-<pid>.fifo
printf F > /tmp/gpu-observer-cupti-<pid>.fifo
```

`S` opens the versioned activity/runtime artifacts and enables CUPTI. `F`
disables both activity kinds, flushes existing records, captures the end clock
pair, and writes the summary.

## Join semantics

Runtime submission chooses the authoritative packed-layout window. The shared
correlation ID then supplies the actual GPU interval. CUPTI timestamps are
affinely normalized from start/end calibration pairs before they are compared
with vLLM's monotonic timestamps.

Compute Sanitizer cannot be loaded in this same process: both it and this agent
consume CUPTI's single subscriber slot on the tested CUDA 13.2 / driver 580
stack. Use explicit timed-CUPTI and deep-SASS modes rather than claiming a
same-run combined trace.
