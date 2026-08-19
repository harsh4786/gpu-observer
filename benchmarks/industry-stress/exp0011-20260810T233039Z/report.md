# EXP-0011 report: BurstGPT fixed-schedule replay on Qwen3-14B

## Outcome

The five-minute BurstGPT window completed correctly in both accepted arms: 271 requests, 144,027 vLLM-reported prompt tokens, 18,915 vLLM-reported completion tokens, and zero request or CUDA errors.

This trace is a realistic mixed-length baseline, not the overload pathology. The clean server had at most 16 running requests, one waiting request, and 25.37% KV-cache usage. The trace therefore exercised continuous batching and long-running decodes without saturating the configured 32 active-sequence limit.

The selected SM121 GEMM block counter executed 9,473,312 callbacks. Against clean, the point differences were -0.14% output throughput, +0.64% average ITL, and +0.55% average request latency. These are single-trace point estimates, not confidence intervals.

## First-principles shape of the experiment

The trace schedules requests for 299 seconds. Each row supplies an arrival time, user-text token length, and requested output length. AIPerf synthesized matching text and issued requests at the recorded relative times. vLLM then continuously combined the requests that were active at each instant.

The benchmark lasted about 365 seconds because sending ended at 299 seconds but 14 long-running requests were still decoding. The final drain is about 66 seconds. Consequently, the maximum 117-second request latency is mainly the trace's 859-token output, not a queue stall.

## Arrival fidelity

AIPerf normalized the first source timestamp, 3,527,100,000 ms, to zero. Relative credit-issuance lag reconstructed from raw records was:

| Arm | Lag p50 (ms) | Lag p95 (ms) | Lag p99 (ms) | Maximum lag (ms) |
|---|---:|---:|---:|---:|
| Clean | 0.137 | 0.970 | 1.298 | 1.481 |
| Selected counter | -0.487 | 0.264 | 0.788 | 1.588 |

Negative values are relative to the first issued credit, not requests sent before the trace. The sub-1.6-ms extrema are negligible beside second-scale inter-arrival gaps, so the client did not materially reshape the workload.

## Corrected token accounting

The first clean replay used AIPerf client-side retokenization. One 387-input/71-output request was reconstructed as 70 output tokens even though its stream contained 71 chunks and `ignore_eos=true`. AIPerf still reported `osl_mismatch_count=0`, so that run is retained under `clean/` but rejected as the correctness baseline.

A one-row preflight with `--use-server-token-count` reported all 71 completion tokens. It also exposed an important boundary:

- BurstGPT/AIPerf user-text length: 387 tokens;
- vLLM prompt usage after the chat template: 395 tokens.

The eight-token difference is the chat wrapper. Across 271 requests, the trace prescribed 141,859 user-text tokens while vLLM processed/reported 144,027 prompt tokens. Both accepted arms reported exactly 18,915 completion tokens.

## Matched clean versus selected counter

| Metric | Clean | Selected counter | Difference |
|---|---:|---:|---:|
| Benchmark duration (s) | 364.817 | 365.319 | +0.138% |
| Output throughput (tok/s) | 51.8479 | 51.7767 | -0.137% |
| Request throughput (req/s) | 0.74284 | 0.74182 | -0.137% |
| TTFT average (ms) | 531.570 | 530.485 | -0.204% |
| TTFT p99 (ms) | 1,140.189 | 1,175.019 | +3.055% |
| ITL average (ms) | 141.229 | 142.132 | +0.639% |
| ITL p99 (ms) | 174.187 | 177.363 | +1.824% |
| Request latency average (ms) | 10,244.287 | 10,300.692 | +0.551% |
| Request latency p99 (ms) | 59,770.032 | 60,258.867 | +0.818% |

The fixed arrival window makes throughput primarily a drain-time metric: the probe added 0.502 seconds to total completion. Average ITL and request latency are the more direct steady-execution signals. The negative TTFT average is noise, not an improvement.

## Device-counter validity

The allowlisted target was:

```text
nvjet_sm121_tst_mma_192x144x64_2_48x72x64_tmaAB_alignCD4_bz_TNNN
```

The final device summary reports:

- 99 CUDA modules seen;
- 1 module patched and 0 patch failures;
- 177,550 total CUDA launches observed;
- 9,600 target-kernel launches;
- 9,473,920 host-expected target block entries;
- 9,473,312 actual device callbacks;
- 0 callback-data failures and 0 drops.

The 608-callback difference exactly repeats EXP-0010 and is consistent with one discovery launch executing before the launch-end callback installs the patch. This repeated invariant is strong evidence, but the collector should still record `discovery_missed_blocks` directly.

## Runtime state

| State metric | Clean | Selected counter |
|---|---:|---:|
| Maximum running requests | 16 | 16 |
| Maximum waiting requests | 1 | 1 |
| Maximum KV-cache usage | 25.37% | 25.37% |
| Maximum GPU temperature | 69 C | 75 C |

Both servers used Qwen3-14B BF16, async vLLM V1 scheduling, FlashAttention 2, and FULL_AND_PIECEWISE CUDA Graphs with an 8-GiB KV cache and `--max-num-seqs 32`. Both were stopped after the run; no compute process remained.

## What this changes in the project plan

1. EXP-0010 is the controller demo workload: it produces a clear queueing pathology at concurrency 64.
2. EXP-0011 is the realism/overhead workload: it preserves production-like mixed lengths and bursts but does not overload this server.
3. Server-reported token counts must be canonical for streamed correctness. Client-side retokenization can be off by one even when generation is correct.
4. The selected block counter is viable under production CUDA Graph replay at this scope. Broad stateful instrumentation remains unsafe.
5. The next semantic experiment should join per-step batch composition to the EXP-0010 overload point, then implement one admission or priority action and measure TTFT p99 recovery.
