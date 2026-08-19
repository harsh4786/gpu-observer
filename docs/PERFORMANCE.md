# Performance and memory contract

These are correctness requirements, not optional cleanup work.

## Enforced invariants

- `observer-core` remains `#![no_std]`.
- `EventRecord` remains pointer-free and no larger than 128 bytes.
- Probe emission performs zero steady-state allocations.
- Probe emission never blocks on a full telemetry queue.
- No mutex, MPSC queue, JSON, name lookup, string formatting, file I/O, or
  syscall is allowed in the inference-facing emission loop.
- Batch semantic events publish all-or-nothing.
- Every loss is externally detectable through sequence gaps.
- Correlation never hides mixed-batch many-to-many relationships.
- Kernel overlap is not double-counted as GPU busy time.

Layout assertions, concurrency tests, and a counting-global-allocator test
enforce the first group of invariants.

## Current memory costs

A ring owns `capacity * 104` bytes of event storage plus small cursor and
allocation metadata.

| Capacity | Event storage |
|---:|---:|
| 4,096 | 416 KiB |
| 16,384 | 1.625 MiB |
| 65,536 | 6.5 MiB |
| 262,144 | 26 MiB |

Capacity must be derived from measured worst-case event rate and the collector's
maximum service gap:

```text
capacity >= next_power_of_two(peak_events_per_second * max_drain_pause_seconds)
```

Do not select a large ring merely because the machine has 128 GB. DGX Spark
unified memory means observer pages compete with model weights, KV cache, CUDA
workspaces, filesystem cache, and GPU-accessed allocations.

The JSON report includes `core_index_bytes`, a lower bound derived from dense
vector capacities. Benchmarks must record peak RSS as well.

## Allocation strategy

Startup:

- allocate and first-touch every ring;
- reserve exact compact record capacity when the count is known;
- use fallible `try_reserve`;
- initialize dictionaries and symbol tables outside tracing.

Steady state:

- reuse fixed records;
- use numeric/interned IDs;
- drain in batches;
- write raw data from collector-owned buffers;
- rotate files outside the inference thread.

Cold correlation sorts records in place. It favors sorted vectors and binary
search over pointer-heavy hash structures. Membership and kernel adjacency are
stored as contiguous index ranges.

## Cache and CPU topology on the development Spark

Measured on 2026-08-06:

- ARM64, 20 online CPUs, one hardware thread per core;
- 64-byte L1 data cache line;
- one NUMA node;
- about 124,609 MiB node memory;
- heterogeneous CPU capacity;
- higher-capacity CPUs: 5-9 and 15-19 at up to 3.9 GHz;
- lower-capacity CPUs: 0-4 and 10-14 at up to 2.808 GHz.

Producer/consumer cursors are therefore padded to 64 bytes. Use a
higher-capacity CPU for the collector drain thread, but do not share it with the
vLLM scheduler or a CUDA submission thread. Record CPU affinity in every
benchmark.

Suggested experiments, not universal defaults:

```bash
taskset -c 15 cargo run --release -p gpu-observer-core \
  --example ring_bench -- 10000000 65536
```

Repeat on multiple higher-capacity cores, report median and tail, and keep the
vLLM/CUDA worker affinity fixed. The system has one NUMA node, so remote-NUMA
placement is not a variable on this Spark; heterogeneous core placement still
is.

## Unified-memory discipline

- Budget observer memory before selecting vLLM KV-cache utilization.
- First-touch ring pages before starting the measured workload.
- Track major/minor faults, process RSS, GPU-visible memory, and system free
  memory.
- Keep raw trace writing sequential and buffered.
- Do not enable `mlock`, huge pages, or aggressive readahead by default;
  benchmark them independently because they can reduce flexibility or steal
  memory from model execution.
- Treat page migration or reclaim as a possible latency cause, not background
  noise.

## Focused token and query capture

Token-level semantic capture is disabled unless a focus request is configured. It uses a startup-sized fixed ctypes buffer and fixed 104-byte semantic records; the producer publishes bounded groups and drops visibly on insufficient ring capacity.

Natural-language query and output capture is a separate frontend cold side channel: at most two nonblocking Unix datagrams, 48 KiB each by default and hard-capped at 1 MiB by the Rust receiver. MessagePack and tokenizer strings never enter the EngineCore semantic loop. The timed benchmark must still measure semantic-only and focused-token modes independently because token conversion in Python is an observer cost.

## Benchmark protocol

Always compare:

1. vLLM without instrumentation;
2. semantic tracing only;
3. host CUDA probes;
4. CUPTI;
5. selected device probes.

Record event rate, drop count, ring high-water mark, CPU time, RSS, page faults,
TTFT, ITL, p50/p95/p99 latency, throughput, and GPU metrics.

Two local SPSC microbenchmark runs observed 50.96-61.33 million events/s and
16.31-19.62 ns/event over five million events with a 65,536-entry ring.
It was unpinned and is only a smoke baseline. It does not establish eBPF,
CUPTI, or end-to-end vLLM overhead.
