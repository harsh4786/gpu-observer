# CUPTI calibration and injection gate

## Outcome

The compatibility gate passed. Exact-version CUPTI 13.2.1 loaded on DGX Spark, exposed API 130201, and produced real device execution records from an externally injected library.

The clock test was substantially better than the first one-shot reading. After eight warm-ups, 64 bracketed samples showed a 16 ns offset span. Best uncertainty was 8 ns in the vLLM runtime image and 24 ns in the developer image. The large negative offset is expected because CUPTI and CLOCK_MONOTONIC use different epochs.

The three-launch CUDA workload produced runtime correlations 126, 128, and 130. The three concurrent-kernel records carried those same IDs, with device durations 1.344 us, 0.928 us, and 0.896 us.

## Interpretation

Clock normalization is not the current weak link for millisecond engine-step bracketing. The required conversion is:

CLOCK_MONOTONIC = CUPTI timestamp + measured offset

For longer experiments, periodically sample pairs and fit an affine mapping rather than assuming an immutable offset.

The NVIDIA injection sample is only a compatibility oracle. It allocates 8 MiB activity buffers and prints records at flush, so it is not acceptable as the low-overhead product path. The production agent still needs bounded buffers, visible drops, fixed records, and asynchronous cold-path persistence.

## Evidence

- calibration-vllm-runtime.txt: exact vLLM runtime calibration.
- calibration-devel-runtime.txt: exact developer runtime calibration.
- injection-vector-add.txt: raw activity and API records.
- injection-summary.txt: correlation-set check.
- source-and-binary.sha256 and injection-gate.sha256: source and executable identities.
