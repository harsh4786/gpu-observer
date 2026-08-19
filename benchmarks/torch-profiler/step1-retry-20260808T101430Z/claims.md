# Evidence ledger

## Direct observations

- OBS-001: Cold, warm, and warm-repeat each produced 378 CUPTI kernel activities.
- OBS-002: Summed kernel time was 11.141 ms cold, 11.016 ms warm, and 11.158 ms warm-repeat.
- OBS-003: The cold live probe produced 296 regular and 85 extended launch records.
- OBS-004: The warm live probe produced 380 regular and one extended launch record.
- OBS-005: Both live probe runs produced 381 records with zero malformed records and zero loss markers.
- OBS-006: Cold computed 44 prompt tokens. Warm requests reused 32 and computed 12.

## Derived results

- DER-001: Exactly 84 launch records migrate from `cuLaunchKernelEx` to `cuLaunchKernel`.
- DER-002: The launch-entry migration exactly explains the original 296 versus 380 count difference.
- DER-003: Prefix caching changed GEMM kernel families but did not materially change summed GPU kernel time for this tiny workload.

## Falsified claims

- FAL-001: "Prefix caching causes 84 additional GPU kernels." This is false for EXP-0001.
- FAL-002: "A cuLaunchKernel-only uprobe gives complete launch coverage." This is false on this stack.

## Supported interpretation

The 44-token cold shape selects NVJet/TMA and tensor-op CUTLASS GEMMs. The
12-token cached shape selects small CUTLASS WMMA GEMMs. Different implementations
enter the driver through different launch APIs, while total GPU work-item count
remains constant.

## Open questions

- Does the same launch-path migration occur for Qwen3-8B?
- Does prefix reuse reduce latency for larger prompt deltas where GEMM work dominates overhead?
- Which dimensions and heuristics select NVJet versus CUTLASS WMMA?
- Does CUDA Graph replay preserve this distinction?
