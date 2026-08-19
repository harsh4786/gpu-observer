# EXP-0008: Compute Sanitizer SASS-instrumentation overhead ladder

## Question

What incremental latency, throughput, memory, and utilization cost is introduced
as Qwen3-14B BF16 observability moves from host subscription to device-side SASS
callbacks?

## Fixed controls

- Hardware: NVIDIA DGX Spark / GB10 / sm_121.
- Model: `Qwen/Qwen3-14B`, revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16.
- vLLM: NGC 26.05 image, V1, TP=1, eager, synchronous scheduling, prefix
  caching off, 4096 maximum context, explicit 8 GiB KV cache.
- Workload: deterministic random-token completions, seed 20260809, 64 requests,
  128 input tokens, exactly 64 output tokens, concurrency 8, all submitted at
  once, temperature zero.
- Warmup: 8 requests of 128 input and 16 output tokens, excluded from results.
- Device event storage: 65,536 records x 64 bytes = 4 MiB per CUDA context.
- Sampled mode: nominal 1/4096 event retention after exact callback counting.

## Common measurement layer

All seven rungs use the same existing minimal vLLM semantic patch and fixed-size
Rust SPSC ring so engine-step begin/end wall time can be measured. Therefore,
`clean` means *no Compute Sanitizer subscriber or SASS patch*, not an absolutely
uninstrumented vLLM process. The earlier EXP-0006 pristine baseline remains the
absolute reference.

Client TTFT, ITL, end-to-end percentiles, and throughput are observed over HTTP
with the client clock. Engine-step wall time is observed by the patched Python
`time.monotonic_ns()` boundaries. Sanitizer callback counts and events are
observed inside patched GPU code. These observation points are not substituted
for one another.

## Rungs

1. `clean`: semantic measurement only; no sanitizer library.
2. `subscriber`: launch/resource subscriber; no SASS patch.
3. `block_noop`: block-entry callback returns immediately.
4. `block_counter`: block leader increments one device counter.
5. `block_event`: block leader reserves and writes one bounded 64-byte event.
6. `sampled_memory_barrier`: exact callback counter plus bounded sampled
   global/shared/local memory and barrier events, gated by earlier results.
7. `full_memory`: every global/shared/local access attempts a bounded event;
   one-request diagnostic stress only, never a serving-performance claim.

For rungs 6 and 7, a kernel substring may defer patching until one selected
Qwen module is identified from rung 2. This selection and the resulting
module-level scope must be recorded. Compute Sanitizer patches a module, not an
individual kernel; non-target functions in that module still contain callback
sites, but receive null callback data and return without being attributed.

## Required outputs

- raw detailed client JSON;
- semantic records plus all/prefill/decode step distributions;
- one-second host/GPU/system samples and their summary;
- per-kernel launches, expected block callbacks, actual callbacks, emitted
  records and drops;
- binary raw device events with raw `%globaltimer` values and SM IDs;
- exact commands, version/source hashes, server logs, and checksums.

## Gates

- Do not advance if the server fails correctness, any request fails, semantic
  records drop, or callback-data setup fails.
- Do not interpret rung 3's actual callback count: the immediate-return callback
  intentionally has no counter. Its expected block count is retained separately.
- Do not run full memory tracing across every Qwen module. Select one repeatedly
  executed production kernel module and bound the event buffer and workload.
- Stop the server after every rung; do not leave the GPU occupied.

