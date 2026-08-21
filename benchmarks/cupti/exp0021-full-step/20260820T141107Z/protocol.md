# EXP-0021 protocol

## Question

Can bounded all-kernel CUPTI tracing explain the GPU-kernel portion of each Qwen3-14B vLLM engine step under production-style asynchronous CUDA Graph execution?

## Fixed factors

- DGX Spark / NVIDIA GB10 / driver 580.173.02
- Qwen/Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`
- BF16, maximum context 4096, 8 GiB KV cache
- vLLM V1 with async scheduling and `FULL_AND_PIECEWISE` CUDA Graphs
- prefix caching disabled
- eight concurrent fixed ShareGPT-derived requests
- temperature 0, EOS ignored, forced output lengths 8 through 36

## Observers

- patched EngineCore semantic events on `CLOCK_MONOTONIC`
- authoritative GPUModelRunner packed-layout events
- CUPTI Runtime and Driver API activity with correlation IDs
- CUPTI concurrent-kernel activity with actual start/end
- two-point affine CUPTI-to-monotonic calibration
- client HTTP/output-token correctness

## Metrics with exact endpoints

- `semantic_wall_ns = semantic_step_end - semantic_step_begin`
- `schedule_to_pack_ns = packed_layout_begin - semantic_step_begin`
- `pack_to_first_api_ns = first_assigned_API_start - packed_layout_begin`
- `api_endpoint_span_ns = last_assigned_API_end - first_assigned_API_start`
- `first_gpu_start_minus_api_end_ns = first_assigned_kernel_start - that_kernel_API_end`
- `kernel_sum_ns = sum(kernel_end - kernel_start)`
- `gpu_busy_union_ns = union of assigned kernel intervals`
- `gpu_span_ns = last_kernel_end - first_kernel_start`
- `gpu_gap_within_span_ns = gpu_span_ns - gpu_busy_union_ns`
- `last_gpu_to_step_end_ns = semantic_step_end - last_kernel_end`

Kernel sum and busy union are intentionally separate.

## Hard rejection gates

Reject on any response mismatch, semantic loss, missing packed layout, failed CUPTI enablement, fixed-buffer exhaustion, output drop, invalid interval, missing API correlation, unassigned kernel, PID mismatch, GPU interval outside its semantic step, or record-accounting mismatch.

A negative GPU-start minus API-end value is retained as valid host/device overlap rather than treated as corruption.

## Warmup and scope

Model load, compilation, CUDA Graph capture, and one warmup request occur before deferred CUPTI start. This is a diagnostic structural trace, not an instrumentation-overhead benchmark. Memcpy, memset, unified-memory, and instruction activities are outside this run.
