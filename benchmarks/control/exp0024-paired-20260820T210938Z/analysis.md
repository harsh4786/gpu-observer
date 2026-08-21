# EXP-0024 — Closed-loop cap on long prefill slices during active decode

## Question

When interactive decodes and a large uncached background prefill would share an engine step, can one explicit scheduler action reduce interactive tail latency without materially harming throughput or background completion?

## First-principles mechanism

The controller changes only one decision inside the Python vLLM scheduler:

```text
active decode request exists
  + candidate prefill slice exceeds 256 tokens
  -> cap that prefill slice to 256 tokens for this engine step
  -> execute the remainder in later steps
```

The action does not treat prefill as a permanent bottleneck. It only bounds the amount of one-time prefill work colocated with active decoding. Arm A used the pristine scheduler with the cap disabled. Arm B used the identical scheduler overlay with `GPU_OBSERVER_PREFILL_CAP_TOKENS=256`.

## Experimental design

- Hardware: NVIDIA GB10 on DGX Spark.
- Model: Qwen/Qwen3-14B BF16, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Runtime: pinned NGC vLLM image, async scheduling, FULL_AND_PIECEWISE CUDA Graphs, prefix caching disabled, 8 GiB KV cache.
- Five paired repetitions, with a fresh server for every arm.
- Frozen arm order: `B A A B A B B A A B`.
- Both arms used the same mixed workload: 100 fixed interactive ShareGPT requests at concurrency 8 with exactly 128 output tokens, plus 45 fixed long-prefill requests injected every four seconds.
- Semantic-only telemetry was used during performance measurement; CUPTI and Compute Sanitizer were absent.
- Every scheduler action emitted the condition, cap, uncapped tokens, and scheduled total to the server log.

The predeclared pass gate required the paired 95% interval to show lower p99 TTFT, throughput loss no worse than 5%, background-duration increase no worse than 25%, and exact action invariants.

## Integrity

All ten arms passed:

- 100/100 interactive requests completed with zero AIPerf errors;
- every interactive request returned exactly 128 output tokens;
- 45/45 background requests reached SSE `[DONE]` in every arm;
- semantic sequence gaps and explicit loss markers were zero;
- thermal aborts were zero and maximum observed GPU temperature was 76 C;
- baseline arms emitted zero policy actions;
- controlled arms emitted 178 or 179 actions;
- every controlled mixed prefill/decode step had a maximum individual prefill slice of exactly 256 tokens;
- every baseline arm exposed a 1,032-token prefill slice in a mixed step.

## Results

| Metric | Baseline mean | Controlled mean | Paired change | 95% paired CI |
|---|---:|---:|---:|---:|
| Interactive p99 TTFT | 676.384 ms | 552.814 ms | **-18.227%** | **-22.051% to -14.403%** |
| Interactive p99 ITL | 136.662 ms | 132.881 ms | **-2.767%** | **-2.947% to -2.587%** |
| Output throughput | 57.132 tok/s | 58.147 tok/s | **+1.778%** | **+1.412% to +2.143%** |
| Mean TTFT | 390.273 ms | 398.107 ms | +2.402% | -10.744% to +15.547% |
| Mean ITL | 132.998 ms | 130.458 ms | -1.910% | -1.979% to -1.840% |
| Benchmark duration | 224.044 s | 220.132 s | -1.746% | -2.099% to -1.393% |
| Background completion window | 178.750 s | 179.177 s | **+0.239%** | **+0.235% to +0.243%** |
| Mixed steps | 64.8 | 238.6 | +268.846% | +248.269% to +289.424% |
| Max prefill slice in mixed steps | 1,032 tokens | 256 tokens | -75.194% | identical in all pairs |

The predeclared gate passes. All five baseline-to-control p99 TTFT changes were negative.

## Why mixed-step count increased

Chunking one 1,032-token prefill into bounded slices deliberately creates more mixed steps. Therefore “number of mixed steps” is not itself a universal pathology indicator. EXP-0023 showed that unbounded collision correlated with worse tails; EXP-0024 shows the important variable is the cost/composition of a mixed step, not merely whether a step is mixed.

## Condition-scoped validator correction

The first offline verdict was `NOT_CONFIRMED` because the analyzer mistakenly enforced `max prefill slice <= 256` across every controlled step. Raw semantic traces showed that the only larger controlled slices—263 or 519 tokens—occurred in prefill-only steps after interactive decoding had ended. They did not satisfy the controller condition and therefore should not have been capped.

The original metrics and verdict are preserved as `*-precondition-fix.*`. The corrected analyzer evaluates the exact predeclared predicate: maximum prefill slice only in steps whose semantic begin record reports decode tokens greater than zero. Reanalysis of the unchanged raw traces gives 1,032 tokens in every baseline mixed step and 256 tokens in every controlled mixed step. No workload was rerun for this correction.

## Interpretation

On this pinned workload, bounded prefill chunking during active decode materially improved interactive tail latency, did not trade away throughput, and added only 0.239% to the background completion window. This is the first complete `observe -> correlate -> diagnose -> act -> measure` result in the project.

The mean TTFT interval crosses zero, so the supported claim is specifically about the tail and ITL, not every latency statistic. The positive throughput result is empirical for this workload, not a guarantee that capping always increases throughput.

## Limits

- Five pairs satisfy the predeclared paired-t gate, but a two-sided exact sign test with five identically directed effects has a floor of p=0.0625.
- One machine, model revision, prompt set, arrival schedule, and cap were tested.
- “Background” is inferred from the long-prefill workload shape; the runtime does not yet carry an explicit service-class label into the scheduler.
- The policy triggers on any active decode and a large prefill. A production controller needs tenant/service-class semantics, starvation bounds, and adaptation across workload intensities.
- Python action logging is included in the controlled arm; its cost was not separately isolated.
- This performance run does not contain CUPTI or instruction-level evidence; those mechanisms were validated separately to avoid observer contamination.

## Evidence map

- `manifest.txt`: frozen environment, policy, workload, and acceptance limits.
- `protocol.sh`: exact executed harness snapshot.
- `run-metrics.tsv`: corrected condition-scoped arm table.
- `paired-summary.tsv`: paired means, changes, intervals, and statistics.
- `paired-analysis.log`: final predeclared PASS verdict.
- `*-precondition-fix.*`: preserved original global-cap analysis.
- `arms/*/semantic.bin`: raw scheduler/packed-layout records.
- `arms/*/semantic-dump.log`: records used to derive mixed-step caps.
- `arms/*/server.log`: structured action evidence.
- `arms/*/metric-summary.json`: raw AIPerf summaries.
- `arms/*/system.csv`: thermal and resource samples.
- `checksums.sha256`: final integrity seal.
