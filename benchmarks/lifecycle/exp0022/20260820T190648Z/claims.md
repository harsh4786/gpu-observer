# EXP-0022 claims

## Accepted

- Twelve EngineCore-accepted token IDs matched the twelve frontend-emitted and
  twelve client-received token IDs exactly.
- All 6,340 physical kernel intervals mapped once to 12 semantic engine steps.
- The measured client TTFT was 155.205 ms and its disjoint causal endpoint
  differences close arithmetically.
- Semantic and CUPTI loss, drop, correlation, assignment, and fatal-error gates
  were all zero.

## Explicitly rejected

- Frontend handler entry is not raw socket arrival.
- Generator yield is not proof of TCP transmit completion.
- The experiment does not include memcpy, memset, UVM migration, or collectives.
- It does not prove per-request ownership inside arbitrary shared kernels.
- It is not an instrumentation-overhead benchmark.
