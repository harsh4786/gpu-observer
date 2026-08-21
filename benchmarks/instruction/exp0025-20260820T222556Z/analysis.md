# EXP-0025 — Bounded request-to-SASS instruction attribution

## Question

Can sampled dynamic instruction events from one production Qwen3-14B BF16 kernel be joined back to the requests whose packed token rows caused that work, without streaming every instruction event?

The tested causal chain was:

```text
request
  -> authoritative post-compaction packed row range
  -> EngineCore step
  -> target CUDA launch ID
  -> blockIdx.x token row
  -> sampled dynamic program counter
  -> exact instruction offset in shipped sm_120 SASS
```

## Scope

This is a diagnostic mechanism experiment, not a latency benchmark and not arbitrary-kernel attribution. The selected kernel is vLLM's production BF16 `reshape_and_cache_flash_kernel`, whose linear `blockIdx.x` maps to packed token rows. Qwen3-14B ran in eager, synchronous mode so each invocation had an ordinary stable launch ID. The workload used eight short requests and one delayed long request, producing prefill, decode, and two mixed steps.

## Instrumentation

- Compute Sanitizer patching instrumented memory operations and barriers in the SASS-only BF16 kernel.
- The callback used a deterministic hash over launch ID, PC, thread, and block and retained approximately one of every 4,096 callbacks.
- Events were fixed 64-byte records in a bounded 1,048,576-record (64 MiB) device buffer.
- The semantic ring carried authoritative request row ranges after `InputBatch` compaction.
- The offline Rust analyzer joined exact launch IDs, checked grid geometry, mapped `blockIdx.x` to an owned packed row, normalized PC to the runtime function base, and required that offset to exist in the extracted `sm_120` SASS.
- All maps and allocations are in the bounded cold analyzer. The inference semantic hot path remains fixed-record, `no_std`, nonblocking, and drop-accounted.

## Integrity and accounting

| Check | Result |
|---|---:|
| Requests completed through SSE `[DONE]` | 9/9 |
| Engine steps | 19 |
| Prefill / mixed / decode steps | 1 / 2 / 16 |
| Semantic sequence gaps / loss markers | 0 / 0 |
| Captured target launch records | 760 |
| Launches with at least one retained sample | 572 |
| Dynamic instruction callbacks in armed window | 59,801,600 |
| Retained samples | 14,363 |
| Effective retained fraction | 0.000240178 (about 1/4,163) |
| Device event drops / sequence errors | 0 / 0 |
| Orphan launch IDs | 0 |
| PC / geometry / ownership mismatches | 0 / 0 / 0 |
| Attributed / padding samples | 14,363 / 0 |
| Fatal CUDA errors | 0 |

The target summary's host-side launch count also includes launches outside the armed event-table window; the causal join deliberately uses the 760 bounded launch records and requires every retained event's launch ID to exist there.

## Dynamic instruction result

All 14,363 retained samples were attributed to one of nine requests. The output contains 144 request-step-phase aggregates: nine prefill aggregates with 13,043 samples and 135 decode aggregates with 1,320 samples.

| Event kind | Samples |
|---|---:|
| Global reads | 10,753 |
| Global writes | 3,610 |
| Shared/local accesses | 0 |
| Barriers | 0 |
| Sampled bytes | 172,664 |

Only five dynamic PC/kind pairs appeared:

| SASS offset | Dynamic event | Samples | Shipped instruction |
|---:|---|---:|---|
| `0x50` | global read | 7,143 | `LDG.E.64.CONSTANT` |
| `0x820` | global read | 1,785 | `LDG.E.128.CONSTANT` |
| `0x870` | global write | 1,785 | `STG.E.128` |
| `0x17a0` | global read | 1,825 | `LDG.E.128.CONSTANT` |
| `0x17f0` | global write | 1,825 | `STG.E.128` |

No barrier callback was observed for this specialization. That is a result about this selected kernel and instrumentation mode, not evidence that transformer execution generally lacks barriers.

## What this earns

For this kernel family, the project now has dynamic evidence below the CUDA-kernel boundary:

```text
request hash + phase + step
  -> packed token row
  -> CUDA block
  -> dynamic load/store PC
  -> exact static SASS instruction
```

The mapping survives the scheduler-order failure found in EXP-0013 because ownership comes from the GPUModelRunner's authoritative post-compaction row layout, not Python scheduler iteration order.

## What this does not earn

- It does not assign arbitrary GEMM or attention tiles to requests. Their block-to-request geometry needs a separate derivation and may be non-identifiable.
- It does not convert samples into per-request nanoseconds, bytes transferred from DRAM, cache misses, or cost.
- Only 572 of 760 launches received a sample; sampling is intentionally sparse and unsuitable for exact per-launch absence claims.
- The 59.8 million callbacks still execute before the 1/4,096 retention decision. No performance claim is made; continuous production use requires instrumentation placement or aggregation that avoids per-instruction callbacks.
- The run is eager and synchronous. Graph-safe block ownership was proven separately in EXP-0018/19, but this exact dynamic-PC experiment was not repeated under graph replay.
- Request IDs are hashed and prompts are not exported through the instruction trace.
- Static SASS classification says what instruction was sampled; it does not reveal higher-level semantic intent by itself.

## Post-processing failure and recovery

The original harness mistook a cuobjdump SASS text-section index for a symbol-table function index and exited after the complete capture. The copied module was independently validated as a normal AArch64 ELF, and the raw semantic, launch, event, response, and summary files were intact.

Without rerunning inference, post-processing was corrected to select the active mangled function name and then isolate the `sm_120` image. The unchanged raw capture passed all joins. `postprocess-recovery.md` records the audit trail, and the reusable runner contains the corrected selector.

## Evidence map

- `manifest.txt`: pinned model, image, hashes, sample rate, buffer bound, and scope.
- `semantic.bin`: raw steps and authoritative packed ownership.
- `device-probe.launches.tsv`: bounded exact launch-ID table.
- `device-probe.events.bin`: raw fixed-record dynamic samples.
- `device-probe.summary.tsv`: callback, emission, drop, module, function-PC, and mode evidence.
- `vllm_C.abi3.so`: exact module copied from the measured container.
- `reshape-and-cache.sass`: isolated shipped `sm_120` disassembly.
- `request-instruction-samples.tsv`: request/step/phase aggregates.
- `pc-instruction-samples.tsv`: exact sampled SASS PCs and instructions.
- `instruction-join.log`: closed accounting and zero-mismatch verdict.
- `postprocess-recovery.md`: rejected cuobjdump-index interpretation and recovery.
- `checksums.sha256`: final integrity seal.
