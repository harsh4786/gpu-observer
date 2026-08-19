# EXP-0010 adaptive branch: concurrency 64

Created after all four registered clean points completed and before any concurrency-64 request.

## Triggering evidence

Clean output throughput increased monotonically through the maximum active-sequence setting: 30.79, 54.04, 87.00, and 121.21 tokens/s at concurrency 4, 8, 16, and 32. A saturation knee was therefore not observed inside the original sweep.

## Added point

Run the identical 1,024-input/128-output, 8-warmup/100-measured AIPerf workload at concurrency 64 on the same clean server.

Because `--max-num-seqs 32`, this point deliberately exceeds execution capacity. The expected mechanism is that at most 32 requests run while the remainder wait, allowing us to distinguish throughput saturation from queue-driven TTFT and request-latency growth.

All original correctness and thermal gates remain in force.

