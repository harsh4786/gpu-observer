# EXP-0004 reproduction protocol

Build cupti-agent/smoke/cupti_timestamp.cpp in the CUDA 13.2.1 developer image. Link against that image's CUPTI headers and runtime, then execute the resulting binary in both the vLLM 26.05 image and the CUDA 13.2.1 developer image.

The calibration program rejects a header/runtime API mismatch, performs eight warm-up reads, then records 64 pairs. Each pair brackets cuptiGetTimestamp with CLOCK_MONOTONIC reads. Offset is midpoint_monotonic minus cupti_timestamp; uncertainty is half the bracket width.

Build NVIDIA's exact cupti_trace_injection sample from the CUDA 13.2.1 developer image. Build cupti-agent/smoke/vector_add.cu for sm_121. Run the unmodified vector-add process with CUDA_INJECTION64_PATH pointing to the sample library.

Accept the gate only if three cudaLaunchKernel runtime activities and three concurrent-kernel activities are present and their CUPTI correlation ID sets are identical.
