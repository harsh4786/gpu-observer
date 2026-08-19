# EXP-0010 adaptive branch: probe comparison at the clean knee

Created after the clean concurrency-64 point and before launching the instrumented server.

## Clean evidence

Output throughput increased from 30.79 to 121.21 tokens/s between concurrency 4 and 32. Raising concurrency to 64 produced only 121.65 tokens/s (+0.36%) while TTFT p99 increased from 12.35 s to 41.85 s and request-latency p99 from 39.45 s to 69.57 s.

Concurrency 32 is therefore the maximum-throughput knee under `--max-num-seqs 32`; concurrency 64 is the queueing overload point.

## Instrumented comparison

Run the identical 8-warmup/100-measured 1,024-input/128-output workload at concurrency 32 only, with the EXP-0009-validated allowlisted block counter.

This avoids spending time repeating low-load points that cannot strengthen the overhead conclusion. The comparison is matched on model, server settings, request generator, seed, concurrency, and token lengths.

