# EXP-0023 — Paired retest of mixed-prefill tail-latency pathology

## Question

Does a frozen stream of long, uncached ShareGPT prefills reproducibly degrade interactive Qwen3-14B latency beyond fresh-server run-to-run variation?

## Design

- Hardware: NVIDIA GB10 on DGX Spark.
- Runtime: pinned NGC vLLM image, Qwen/Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16, production async scheduling, FULL_AND_PIECEWISE CUDA Graphs, prefix caching disabled.
- Five paired repetitions; every arm used a fresh server.
- Arm order was randomized once with seed 20260820 and frozen before execution: `B A A B B A A B B A`.
- Arm A: 100 fixed interactive requests, concurrency 8, exactly 128 output tokens each.
- Arm B: the same interactive workload plus 45 fixed long-prefill requests injected every four seconds, starting 0.5 seconds after AIPerf profiling began.
- Performance runs used semantic-only instrumentation. CUPTI/Sanitizer mechanism experiments are separate so profiler overhead cannot contaminate the performance claim.
- Primary inference: paired, run-level, two-sided t 95% interval over within-pair percentage changes. The predeclared pass condition required positive lower confidence bounds for p99 TTFT and mixed-step count.

The earlier partial pilot at `../exp0023-paired-20260820T191627Z` is explicitly rejected because compilation ran concurrently with measured arms.

## Integrity

All ten arms passed:

- AIPerf exit status 0 and no request errors;
- exactly 100 measured interactive responses per arm;
- exactly 128 output tokens per interactive response;
- all 45 background requests reached SSE `[DONE]` in every B arm;
- semantic sequence gaps: 0 in every arm;
- semantic loss markers: 0 in every arm;
- maximum in-flight engine steps: 2 in every arm;
- thermal aborts: 0;
- observed GPU temperature: 45–74 C, below the predeclared 85 C abort threshold.

## Results

| Metric | Arm A mean | Arm B mean | Paired change | 95% paired CI |
|---|---:|---:|---:|---:|
| p99 TTFT | 429.557 ms | 684.117 ms | +59.220% | +34.768% to +83.671% |
| p99 ITL | 124.791 ms | 136.497 ms | +9.380% | +9.013% to +9.747% |
| Output throughput | 60.956 tok/s | 57.232 tok/s | -6.109% | -6.339% to -5.878% |
| Mean ITL | 124.255 ms | 132.607 ms | +6.722% | +6.657% to +6.786% |
| Mixed steps | 13.8 | 64.8 | +369.780% | +343.121% to +396.440% |

Every pair had the same direction:

| Pair | p99 TTFT delta | p99 ITL delta | Throughput delta | Extra mixed steps |
|---:|---:|---:|---:|---:|
| 1 | +243.297 ms | +11.449 ms | -3.885 tok/s | +50 |
| 2 | +269.415 ms | +11.683 ms | -3.756 tok/s | +50 |
| 3 | +247.592 ms | +11.735 ms | -3.772 tok/s | +50 |
| 4 | +374.753 ms | +12.314 ms | -3.600 tok/s | +56 |
| 5 | +137.748 ms | +11.349 ms | -3.607 tok/s | +49 |

## Interpretation

Under this frozen workload, long-prefill interference is reproducibly associated with many more mixed prefill/decode steps and materially worse interactive tails. The predeclared EXP-0023 criterion passes.

This experiment establishes the serving pathology and its scheduler-level collision signature. It does not, by itself, prove how the additional cost divides among individual kernels; EXP-0021 supplies the complete GPU budget mechanism, while EXP-0025 tests one kernel-internal attribution path.

## Statistical limits

Five pairs are enough for the predeclared paired-t gate, but the t interval relies on the distribution of paired effects. All five TTFT effects are positive; nevertheless, an exact two-sided sign test with only five pairs has minimum p=0.0625. Therefore the correct claim is “reproducible in all five paired runs and passes the predeclared paired-t criterion,” not a universal production guarantee. A publication-quality follow-up should use more independent pairs and preferably a blocked analysis across multiple workload intensities.

## Confounds controlled and remaining

Controlled:

- same model/revision/image and workload files;
- fresh server for every arm;
- randomized frozen order;
- fixed request count, concurrency, output length, and background schedule;
- no CUPTI, Nsight, Sanitizer, compilation, or analysis during the accepted run;
- explicit semantic loss and thermal gates.

Remaining:

- one machine and one model revision;
- five pairs;
- Arm B changes both offered load and batch composition, so EXP-0023 proves the workload pathology, not an isolated kernel-level causal coefficient;
- prefix caching was disabled intentionally, so the result applies to uncached prefills;
- background throughput is not a primary output of this experiment.

## Evidence map

- `manifest.txt`: frozen environment and protocol.
- `protocol.sh`: executed harness copy.
- `run-metrics.tsv`: arm-level measurements.
- `paired-summary.tsv`: paired means, changes, intervals, and t statistics.
- `paired-analysis.log`: predeclared verdict.
- `arms/*/metric-summary.json`: raw AIPerf summaries.
- `arms/*/semantic.bin`: raw semantic records.
- `arms/*/semantic-step-stats.tsv`: loss and step-class checks.
- `arms/*/system.csv`: raw thermal/resource samples.
- `checksums.sha256`: integrity seal.
