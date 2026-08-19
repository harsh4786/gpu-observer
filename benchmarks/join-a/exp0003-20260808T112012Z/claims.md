# EXP-0003 claims

## Supported

- One request was represented by one stable 64-bit request hash across one prefill and two decode engine steps.
- Each step contained a dense begin, request-slice, end sequence with no semantic loss.
- All 1,146 external CUDA launch records fell within exactly one engine-step interval.
- The prefill step used 85 extended launch calls; each decode step used one. This reproduces the entry-point-selection pattern found in EXP-0001.
- Launches came from one TID in this run, but the wire format still records TID.
- Container PID 300 and host PID 273358 refer to the same EngineCore and must be normalized explicitly.

## Not supported yet

- Actual GPU kernel start, end, duration, overlap-safe busy time, or GPU idle gaps. Those require CUPTI activity.
- Per-request attribution inside a mixed-batch kernel.
- Correct semantic coverage under asynchronous scheduling.
- Performance conclusions from one cold, non-streaming request on Qwen3-0.6B.
