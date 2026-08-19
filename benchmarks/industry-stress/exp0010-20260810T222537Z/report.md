# EXP-0010 report: Qwen3-14B production saturation and selected device-counter overhead

## Outcome

On DGX Spark, Qwen3-14B BF16 reached its useful throughput knee at concurrency 32 under the configured `--max-num-seqs 32`. Raising offered concurrency to 64 increased output throughput only 0.36%, from 121.21 to 121.65 tokens/s, while client-visible queueing drove TTFT p99 from 12.35 to 41.85 seconds and request-latency p99 from 39.45 to 69.57 seconds.

At concurrency 32, a Compute Sanitizer block-entry counter restricted to one hot SM121 NVJITLINK GEMM produced 3,205,152 device callbacks. Its output-throughput point estimate was 119.89 tokens/s versus 121.21 clean, a -1.09% difference. This is a useful first overhead bound, not a confidence interval: each arm is one 100-request run separated by a server restart.

## Clean saturation sweep

Each point used 8 warmup requests followed by 100 measured streaming requests with exactly 1,024 input and 128 output tokens.

| Concurrency | Output tok/s | Request/s | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) | ITL p99 (ms) | Request latency avg (ms) | Request latency p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 30.794 | 0.2406 | 1,011.10 | 1,639.70 | 122.87 | 126.06 | 16,616.10 | 16,831.68 |
| 8 | 54.038 | 0.4222 | 1,380.37 | 2,945.59 | 133.41 | 139.98 | 18,322.86 | 18,921.33 |
| 16 | 87.000 | 0.6797 | 2,298.22 | 5,919.57 | 152.93 | 166.22 | 21,720.53 | 25,195.58 |
| 32 | 121.208 | 0.9469 | 3,796.00 | 12,348.23 | 203.54 | 227.18 | 29,646.14 | 39,449.85 |
| 64 | 121.649 | 0.9504 | 24,948.42 | 41,847.01 | 206.35 | 227.22 | 51,154.53 | 69,571.17 |

The mechanism is visible in the metric shape: concurrency 64 cannot increase active execution beyond 32 sequences, so throughput stays flat while excess requests wait. TTFT absorbs almost all of the new delay; ITL p99 is essentially unchanged (227.18 versus 227.22 ms). This is queueing saturation, not a slower decode kernel.

## Matched clean versus selected counter at concurrency 32

| Metric | Clean | Selected counter | Difference |
|---|---:|---:|---:|
| Output throughput (tok/s) | 121.2079 | 119.8855 | -1.091% |
| Request throughput (req/s) | 0.94694 | 0.93661 | -1.091% |
| TTFT average (ms) | 3,795.996 | 3,887.939 | +2.422% |
| TTFT p99 (ms) | 12,348.227 | 12,238.107 | -0.892% |
| ITL average (ms) | 203.544 | 205.177 | +0.802% |
| ITL p99 (ms) | 227.176 | 230.268 | +1.361% |
| Request latency average (ms) | 29,646.142 | 29,945.403 | +1.009% |
| Request latency p99 (ms) | 39,449.853 | 39,384.797 | -0.165% |

Negative p99 differences are run noise, not an instrumentation improvement. The coherent signals are the -1.09% throughput and approximately +0.8% to +1.0% average token/request latency changes.

## Device-counter validity

The accepted probe target was:

```text
nvjet_sm121_tst_mma_192x144x64_2_48x72x64_tmaAB_alignCD4_bz_TNNN
```

The final summary reports:

- 99 CUDA modules seen;
- 1 module patched and 0 patch failures;
- 6,640 target launches;
- 3,205,760 host-expected block entries;
- 3,205,152 device callbacks;
- 0 event drops and 0 callback-data failures.

The 608-count shortfall is consistent with the first matching launch executing before its launch-end callback discovers and patches the module. The current aggregate format does not record that first launch's grid independently; add an explicit `discovery_missed_blocks` counter before treating this invariant as self-proving.

The probe remained active through production FULL_AND_PIECEWISE CUDA Graph capture and replay. All 100 measured responses contained exactly 128 output tokens, and the final server log contains no fatal CUDA error.

## Workload matching and the rejected arm

The first attempted counter arm is retained under `selected_counter_c32/` but excluded from overhead claims:

- its requested kernel never executed (`modules_patched=0`);
- AIPerf regenerated a different synthetic prompt corpus despite the same seed.

For the corrected arm, AIPerf loaded the clean `inputs.json` through its raw-payload replay loader. AIPerf's regenerated output artifact contains empty message content and therefore has a different file hash—an artifact-export defect also reflected by a skipped upstream replay integration test. The traffic itself passed two independent matching checks:

- all 100 session IDs had exactly matching HTTP request-body sizes, request by request;
- both vLLM server-metric captures reported 103,200 prefill tokens.

This supports equal workload execution, but future publication runs should hash the serialized body immediately before transport to prove byte identity directly.

## Execution and safety evidence

- Model: `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16.
- vLLM: NGC 26.05 build `0.20.1+7124b12a.dev`.
- Execution: asynchronous production path, FlashAttention 2, FULL_AND_PIECEWISE CUDA Graph mode.
- Limits: context 8,192, KV cache 8 GiB, maximum 32 active sequences.
- Maximum recorded GPU temperature: 81 C clean and 75 C during the corrected probe run; the registered abort threshold was 85 C.
- Every server was stopped after its arm. Final state: no compute application, 0% GPU utilization, 45 C.

## Interpretation

This experiment establishes two useful facts for the hackathon:

1. A reproducible overload condition now exists: offering 64 concurrent requests to a 32-sequence server preserves throughput but inflates TTFT p99 by 3.39x. That is an excellent target for the future detector and admission controller.
2. Selective SASS-level instrumentation survives the real Qwen3-14B CUDA Graph path at roughly a 1% throughput point cost. All-module stateful instrumentation remains unsafe from EXP-0009.

It does not yet establish statistical significance for a 1% overhead delta, request-to-block attribution, or a closed-loop policy improvement.
