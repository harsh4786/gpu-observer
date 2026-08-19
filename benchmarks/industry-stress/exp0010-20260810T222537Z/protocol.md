# EXP-0010: AIPerf production saturation sweep

## Question

Where is the Qwen3-14B throughput/latency saturation knee on DGX Spark, and what overhead does one allowlisted block-counter probe add under the same production CUDA Graph path?

## Workload

AIPerf v0.10.0 sends deterministic streaming synthetic chat requests with exactly 1,024 input tokens and 128 output tokens. Each load point has 8 warmup requests and 100 measured requests. Concurrency sweeps over 4, 8, 16, and 32.

This is an industry-tool controlled saturation benchmark, not an MLPerf result. The 1k/128 shape is an interactive-serving qualification shape; it intentionally differs from the more expensive canonical 1k/1k throughput shape.

## Arms

1. Clean NGC vLLM production path.
2. The same path plus Compute Sanitizer subscriber and a block counter allowlisted to one observed sm_121 NVJITLINK GEMM kernel.

No eager flag or synchronous-scheduler override is allowed. vLLM logs must establish async scheduling and FULL_AND_PIECEWISE CUDA Graph capture.

## Observer boundaries

- AIPerf owns client TTFT, ITL, request latency, throughput, token counts, and percentiles.
- vLLM logs own backend, scheduler, graph-mode, and fatal-error evidence.
- Compute Sanitizer owns launch geometry and device callback counts.
- nvidia-smi provides coarse hardware state only; it is not kernel timing.

## Correctness gates

- exactly 100 valid requests and zero errors per concurrency;
- every response must contain exactly 128 output tokens;
- no EngineCore fatal or CUDA error in server logs;
- selected device callbacks must equal all blocks after the first discovery launch;
- no GPU compute process may remain after an arm.

## Interpretation limits

Only differences between matched arms and load points are attributable. Startup/capture work is excluded from AIPerf timing but remains visible in device totals until post-workload flush. If the selected probe is not exercised during measured traffic, no overhead claim is permitted.

