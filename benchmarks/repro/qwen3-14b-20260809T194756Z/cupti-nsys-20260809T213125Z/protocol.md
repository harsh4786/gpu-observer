# EXP-0006 reproduction protocol

## Causal question

Determine whether the interval between Aya's final CUDA launch submission and the patched vLLM engine-step end is actual asynchronous GPU execution or host/framework delay.

## Observation points

- Patched vLLM emits engine-step begin/end using `CLOCK_MONOTONIC`.
- Nsight/CUPTI emits runtime API and device kernel start/end relative to the trace session origin.
- Client-side monotonic/realtime pairs translate the Nsight session origin into the semantic clock domain.

## Procedure

1. Start Qwen3-14B BF16 under delayed Nsight collection with eager execution, synchronous scheduling, prefix caching disabled, an 8 GiB KV cache, and max model length 4096.
2. Wait for health while collection is paused.
3. Issue the deterministic 24-prompt-token, 3-output-token request once as warm-up and drain its semantic records.
4. Start a fresh semantic consumer and the delayed Nsight session.
5. Record a monotonic/realtime pair, issue the request exactly once, record a second pair, and stop Nsight immediately.
6. Export `.nsys-rep` and SQLite, stop the server, normalize kernel timestamps, and assign each kernel by start time to its enclosing semantic step.

Run the checked orchestrator:

```bash
cd /home/harsh4786/gpu-observer
./repro/qwen3-14b/capture-cupti-nsys.sh
```

Analyze the timestamped output directory:

```bash
./repro/qwen3-14b/analyze-cupti-nsys.py \
  /home/harsh4786/gpu-observer/benchmarks/repro/qwen3-14b-20260809T194756Z/cupti-nsys-20260809T213125Z
```

Reject a run if semantic records are incomplete, any kernel is unassigned, any kernel lacks a runtime correlation, clock drift is not small relative to step duration, or either raw Nsight artifact is empty.

