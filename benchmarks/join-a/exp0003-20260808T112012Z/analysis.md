# EXP-0003 analysis

This is the first complete request-to-engine-step-to-CUDA-submission path on the machine.

The request produced three model iterations: 39-token prefill, then two one-token decode steps. The same request hash appears in every slice. Temporal bracketing assigned 382 launches to each step, with no leftovers and no loss indicators.

The unusually long second step is evidence only about wall time and host submission span. It is not yet a GPU-duration result. The next instrumentation step is CUPTI API and kernel activity so submission records gain correlation IDs and actual device start/end times.

Two corrections are now part of the architecture: async and synchronous vLLM paths need separate semantic handling, and process identity must include PID namespace mapping rather than a bare integer.
