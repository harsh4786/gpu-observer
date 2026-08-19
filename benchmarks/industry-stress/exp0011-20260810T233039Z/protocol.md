# EXP-0011: BurstGPT P90 five-minute fixed-schedule replay

## Question

How does Qwen3-14B behave under a real mixed-length burst trace, and does one allowlisted production GEMM block counter preserve correctness and goodput under the same arrival schedule?

## Trace

The pinned input contains 271 valid requests from BurstGPT timestamps [3527100, 3527400) seconds. AIPerf converts seconds to milliseconds and `--fixed-schedule-auto-offset` shifts the first request to time zero while preserving every inter-arrival gap.

The trace prescribes 141,859 input tokens and 18,915 output tokens over 299 seconds. Input lengths range up to 2,431 tokens and output lengths up to 859 tokens.

## Server controls

Both arms use Qwen3-14B BF16, async vLLM V1 scheduling, FlashAttention, FULL_AND_PIECEWISE CUDA Graphs, 8,192 maximum context, 8 GiB KV cache, and 32 active sequences. The AIPerf concurrency ceiling is 271 so the client does not reshape the trace; vLLM owns queueing.

## Arms

1. Clean NGC vLLM production path.
2. The same path plus the EXP-0010-selected SM121 GEMM block counter.

## Observer boundaries

- AIPerf owns scheduled arrivals, dispatch lag, TTFT, ITL, request latency, throughput, and goodput.
- vLLM metrics own queue depth, active requests, KV-cache use, and token totals.
- Compute Sanitizer owns module-patch, launch, block-callback, and drop counts.
- nvidia-smi provides coarse thermals and power, not kernel timing.

## Correctness and safety gates

- exactly 271 completed requests and zero AIPerf errors;
- exactly 18,915 output tokens in aggregate with every per-request OSL matching the trace;
- no fatal CUDA or EngineCore error;
- no client concurrency throttling of scheduled arrivals;
- selected arm requires at least one patched module and nonzero callbacks;
- abort at 85 C;
- no GPU compute process after each arm.

## Interpretation limits

This is a single fixed trace window, not a population confidence interval. A clean-versus-probe difference is a point estimate. Queueing and burst recovery are causal only if AIPerf dispatch lag remains bounded; otherwise the client altered the offered arrival process.
