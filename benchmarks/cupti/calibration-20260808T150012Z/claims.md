# Claims

Supported:

- CUPTI header API 130201 exactly matches the runtime in the vLLM 26.05 image.
- The measured CUPTI-to-CLOCK_MONOTONIC offset had a 16 ns span across 64 warmed samples.
- CUDA injection works on this GB10 stack without modifying the CUDA application.
- CUPTI correlation IDs exactly joined three runtime launches to three GPU kernel executions.

Not supported yet:

- Long-duration clock drift is negligible.
- The NVIDIA sample has acceptable vLLM overhead or bounded memory use.
- Injection into the complete vLLM process is production-ready.
