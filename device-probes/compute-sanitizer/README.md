# Compute Sanitizer device probe

This component uses NVIDIA's public Compute Sanitizer API to subscribe to CUDA
launch/resource events and, in patching modes, insert callbacks into already
compiled SASS. It does not require PTX.

For matched deep replay, each selected function row also records the official `sanitizerGetFunctionPcAndSize` base PC and byte size. The cold exporter rejects global or per-function drops, zero target callbacks, callback/emitted/retained count disagreement, events from unselected functions, out-of-range PCs, ambiguous or empty SASS sections, duplicate offsets, and any sampled offset absent from the selectively dumped function disassembly.

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

Set `GPU_OBSERVER_SAN_ARM_ON_SIGNAL=1` for the replay harness. The first `SIGUSR2` after graph warmup resets and arms bounded device capture; the second requests a flush on the next launch. This excludes graph-capture warmup callbacks from the measured replay.

CUDA Graph mode uses a different identity path. Set
`GPU_OBSERVER_SAN_CALLBACK_DATA_SCOPE=function` to bind one stable device
callback pointer to the patched function; do not rewrite callback userdata for
each replay. With `GPU_OBSERVER_SAN_GRAPH_NODES=1` and a required kernel
substring, the subscriber records graph node-begin callbacks in
`*.graph-nodes.tsv`. A replayed node is identified by the tuple
`(graph_exec, graph_launch_id, node)` while the node handle remains stable
across replays. These are bounded host callback records: the fixed table is 6
MiB (`65,536 * 96` bytes), overflow is counted, and the callback does not
allocate.

With function-scoped callback data, set
`GPU_OBSERVER_SAN_PASSIVE_LAUNCHES=1` to also record ordinary target-kernel
launch-begin callbacks in `*.launches.tsv`. This bounded table is used to prove
that a selected graph-replay suffix contains no eager launches that would make
ordered device-event partitioning ambiguous. It is mutually exclusive with
per-launch callback identity, has a fixed `GO_SAN_MAX_LAUNCHES` capacity, and
reports overflow through `launch_table_drops`.

The graph node timestamp is host `CLOCK_MONOTONIC` at the sanitizer callback;
it is not device kernel start time. An ordered node/event join is valid only
after proving single-stream execution and complete geometry. General
multi-stream attribution must join device `%globaltimer` events to CUPTI kernel
activity intervals.

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

