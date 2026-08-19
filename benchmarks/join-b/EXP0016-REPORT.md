# EXP-0016 — Production CUDA Graph SASS block-event compatibility gate

## Outcome

**Fail closed.** The selected Compute Sanitizer block-entry event patch for
`reshape_and_cache_flash_kernel` is not safe under this vLLM CUDA Graph replay
configuration.

Two independent identity modes reached the same device fault:

| Attempt | Callback identity | Result |
|---|---|---|
| `exp0016-production-gate-20260812T220424Z` | Per-launch userdata (`launch_identity=1`) | CUDA misaligned-address fault |
| `exp0016-production-function-event-20260812T221038Z` | Function-scoped userdata (`launch_identity=0`) | CUDA misaligned-address fault |

Both attempts used Qwen3-14B BF16, production asynchronous scheduling,
FULL_AND_PIECEWISE CUDA Graphs, a 1,048,576-record fixed event buffer, and the
same allowlisted cache kernel.

## What happened

Short requests began executing successfully. When the graph-replayed workload
continued, EngineCore surfaced:

```text
torch.AcceleratorError: CUDA error: misaligned address
```

The error became visible at
`self.async_copy_ready_event.synchronize()`. CUDA reports asynchronous device
faults at a later synchronization point, so that Python line is where the fault
was observed, not proof that the copy event caused it. The instrumented device
execution before that synchronization is the suspect boundary.

The long request received HTTP 500 in both attempts, and each
`request-exit-status.txt` is 1. These are rejected runs, not performance data.

## Why the second attempt matters

Per-launch userdata is naturally risky with graph replay because capture and
replay decouple the original launch call from repeated device execution. The
function-scoped attempt removed that changing launch identity. It still faulted.

Therefore the evidence rejects a narrow explanation that only the per-launch
identity table was unsafe. It does not identify the exact SASS transformation
or callback instruction responsible.

## What remains valid

- The same selected block-entry probe works in eager mode (EXP-0014/0015).
- Compute Sanitizer can patch the production SASS-only BF16 kernel.
- A selected GEMM counter worked under graphs in EXP-0009/0010.
- Compatibility is therefore kernel-, callback-, and graph-mode-specific; it
  cannot be generalized from one successful allowlist.

## Decision

Do not load this cache-kernel block-event sensor in the production demo. Keep
production attribution honest at:

```text
request -> scheduler step -> authoritative packed rows
```

Then use CUPTI/NVTX for graph-safe step-to-kernel timing. Device-block ownership
remains an eager diagnostic feature until a new graph-safe probe passes this
gate.

## Unknown requiring a separate experiment

A minimal matrix is needed to isolate the fault:

1. block callback that immediately returns;
2. callback reading `blockIdx` without storing;
3. function-scoped aligned device counter;
4. fixed event write without launch identity;
5. the same rungs on this exact cache kernel with graph capture and replay.

Until that matrix passes, changing buffer alignment or suppressing the reported
synchronization error would be confirmation-biased debugging.
