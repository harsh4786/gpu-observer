# EXP-0006 Workload A analysis

The clean Qwen3-14B eager smoke baseline completed correctly: one measured request, zero failures, 256 input tokens, and 64 output tokens.

## Observed figures

| Metric | Value |
|---|---:|
| Time to first token | 171.428 ms |
| Mean time per output token | 122.359 ms |
| Mean inter-token latency | 122.359 ms |
| p50 inter-token latency | 122.167 ms |
| p95 inter-token latency | 124.049 ms |
| p99 inter-token latency | 125.184 ms |
| End-to-end latency | 7,880.072 ms |
| Output throughput | 8.122 tokens/s |

The 7.88-second end-to-end duration is consistent with a 171 ms first-token delay followed by 63 inter-token intervals averaging about 122.36 ms.

## Interpretation boundary

This is a correctness and single-request decode-rate measurement, not a tail-latency result. Request-level p50, p95, and p99 all collapse to the same observation because `n=1`. The inter-token percentiles summarize 63 intervals, but all belong to the same request and are therefore not independent request samples.

No attribution or causal claim is made here. The value of this run is that it fixes the model, server controls, client protocol, token counts, and raw-result format before concurrency is introduced.

## Next experiment

Run continuous decode with fixed short prompts, fixed long outputs, and bounded concurrency. Repeat the measured workload three times without restarting the server, preserving each raw result independently. That establishes the first request-level latency distribution and tests baseline stability.
