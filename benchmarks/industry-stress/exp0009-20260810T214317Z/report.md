# EXP-0009 report: production CUDA Graph compatibility gate

## Outcome

The clean server, subscriber-only arm, and all-module block-no-op arm completed the identical 128-input/16-output AIPerf request. Both all-module block-counter variants failed after one output token with `CUDA error: misaligned address`: the original per-launch callback-data path and an experimental function-scoped path.

A selective block counter on `nvjet_sm121_tst_mma_160x128x64_2_80x32x64_tmaAB_alignCD4_bz_TNNN` completed correctly under asynchronous scheduling and FULL_AND_PIECEWISE CUDA Graph execution.

## Controlled evidence

| Arm | Exact OSL | TTFT (ms) | ITL (ms) | Request latency (ms) | Server result |
|---|---:|---:|---:|---:|---|
| clean | 16 | 251.68 | 115.32 | 1981.41 | pass |
| subscriber only | 16 | 252.30 | 115.81 | 1989.41 | pass |
| all-module block no-op | 16 | 253.54 | 115.94 | 1992.70 | pass |
| all-module block counter, launch-scoped data | 1 | n/a | n/a | 255.72 | fatal misaligned address |
| all-module block counter, function-scoped data | 1 | n/a | n/a | 250.14 | fatal misaligned address |
| selected sm_121 GEMM block counter | 16 | 258.91 | 116.02 | 1999.27 | pass |

All AIPerf `inputs.json` files had SHA-256 `e6c1f8f55e2b3cefb122496fd50fa2446915fc807473a0d7c274d211a4e67b8d`.

These are one-request compatibility figures, not statistically valid overhead estimates.

## Device-count invariant

The selected kernel appeared in 40 launches with 360 blocks per launch: 14,400 host-expected block entries. The device callback counted 14,040, exactly 39 x 360. The first matching launch is intentionally missed because its launch-end callback is what discovers and patches the module. Every subsequent launch was counted.

The summary also recorded 98 modules seen, one module patched, zero module-patch failures, zero callback-data failures, and zero setup-missed launches.

## Causal interpretation

1. Subscriber-only passing rules out the host subscriber as the cause.
2. All-module no-op passing rules out SASS insertion and callback control flow by themselves.
3. Function-scoped callback data failing rules out the simple per-launch lifetime hypothesis.
4. Selected counter passing proves counter instrumentation can coexist with production CUDA Graph replay when restricted to a validated kernel family.
5. Therefore the unsafe condition is heterogeneous all-module stateful instrumentation; the exact incompatible module or kernel remains unidentified.

## Tooling correctness finding

AIPerf reported the truncated one-token streams as one valid request with zero errors, and the failed container eventually exited with status zero. Future gates must require exact output length plus a server fatal-error scan; HTTP success and process exit status are insufficient.

## Observer boundaries

- AIPerf measured client-visible token counts and latency.
- vLLM logs established async scheduling, FlashAttention 2, and FULL_AND_PIECEWISE CUDA Graph mode.
- Compute Sanitizer launch callbacks counted host-visible launches and grid dimensions.
- The inserted device callback counted block-entry executions.
- The asynchronous vLLM synchronization line only surfaced an earlier device fault; it did not identify the kernel that caused it.

## Decision

Proceed to load testing with clean, subscriber-only, and allowlisted selected-kernel probe arms. Do not run stateful all-module instrumentation in production mode. Keep the all-module crash as a diagnostic branch and isolate the incompatible kernel family separately.

