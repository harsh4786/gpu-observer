# EXP-0005 reproduction protocol

## Purpose

Measure actual GPU execution time for each semantic engine step and determine what host and device visibility changes when eager execution is replaced by the vLLM default compiled CUDA Graph path.

## Common server controls

Use Qwen/Qwen3-0.6B, BF16, max model length 4096, an 8 GiB KV cache, TP=1, and synchronous scheduling. Disable prefix caching for the corrected runs. Mount the fixed semantic adapter and native bridge. Use the identical request in request.json: 39 prompt tokens, 3 completion tokens, temperature zero, non-streaming.

## Bounded collection

Launch the server through Nsight Systems with collection delayed. Disable CPU sampling and context-switch tracing. Enable CUDA and NVTX only, with CUDA Graph node granularity. Wait for health before any request.

Issue one request while collection is paused to warm lazy modules and the selected execution path. Drain those nine semantic records into warmup-semantic.bin. Start a fresh semantic consumer and the delayed Nsight session, issue the same request once, stop Nsight immediately, and let the semantic consumer flush.

Repeat for eager mode with enforce-eager and for graph mode without enforce-eager. In both corrected runs pass no-enable-prefix-caching and no-async-scheduling.

## Clock join

Nsight SQLite activity times are relative to TARGET_INFO_SESSION_START_TIME.utcEpochNs. Record CLOCK_MONOTONIC and realtime immediately before and after the request. Convert the session origin to monotonic using the per-run offset:

session_origin_monotonic = utcEpochNs + (monotonic_ns - realtime_ns)

Then add each relative CUPTI activity timestamp. Assign a kernel to the semantic step containing its start timestamp. Reject the trace if any kernel lies outside all three measured steps.

Report both summed kernel duration and overlap-safe union duration. Do not use host API duration as GPU time.

## Validation

Require nine semantic records, three complete steps, zero semantic drops, identical request and output token counts, complete runtime-to-kernel CUPTI correlations, and total per-step kernel counts equal to the trace total. Preserve both the cold first-request captures and the corrected warmed captures.
