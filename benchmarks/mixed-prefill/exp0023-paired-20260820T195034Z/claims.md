# EXP-0023 claim boundary

## Supported

For this pinned Qwen3-14B workload on GB10, adding 45 fixed long uncached prefills increased interactive p99 TTFT by 59.220% (95% paired-t CI 34.768–83.671%), increased p99 ITL by 9.380% (CI 9.013–9.747%), reduced output throughput by 6.109%, and increased mixed prefill/decode steps from 13.8 to 64.8 on average. All five pairwise TTFT effects were positive, all ten arms passed request and trace integrity checks, and no arm exceeded 74 C.

## Not supported

- The result is not yet generalized beyond this machine, model, prompt set, or load level.
- The result does not assign the extra latency to individual GPU instructions or arbitrary kernels.
- Five pairs do not support a strong distribution-free significance claim; the exact two-sided sign-test floor is p=0.0625.
- The experiment does not show that a controller improves the outcome; that is EXP-0024.
- The experiment does not measure production tracing overhead beyond semantic-only instrumentation.
