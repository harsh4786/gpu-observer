# EXP-0009: Production-path compatibility gate

## Question

Can the exact Qwen3-14B BF16 production execution path run correctly through
AIPerf first without instrumentation and then with the Compute Sanitizer
block-counter patch?

## Causal comparison

The two arms hold model, request, vLLM configuration, client, and hardware
constant:

1. clean CUDA Graph execution;
2. Compute Sanitizer subscriber plus one block-leader counter increment.

This is a compatibility gate, not a performance benchmark.

## Request

AIPerf v0.10.0 sends one streaming OpenAI chat request with a deterministic
128-token synthetic input and exactly 16 output tokens. EOS is ignored and
temperature is zero.

## Observer boundaries

- AIPerf measures client-side TTFT, ITL, request latency, token counts, and
  correctness of the streaming response.
- vLLM logs establish scheduler mode and CUDA Graph configuration.
- Compute Sanitizer host callbacks establish modules, launches, and grids.
- The inserted device callback counts block-leader executions.
- Host and NVIDIA tools record process topology, utilization, temperature, and
  final idle state.

No timing value from one boundary is substituted for another.

## Gates

Stop before long stress tests if:

- either request fails or produces the wrong output length;
- CUDA Graph execution is silently disabled;
- the Sanitizer arm reports a patch or callback-data failure;
- expected block counts disagree with device callback counts;
- the server cannot stop cleanly or leaves a GPU compute process.

The existing semantic patch targets the synchronous EngineCore.step path. This
experiment records whether production async scheduling is active, but does not
pretend that missing semantic events imply no engine steps occurred.
