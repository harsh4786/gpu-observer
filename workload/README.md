# Workload

Deterministic vLLM request generation belongs here. The workload process is
outside inference and may use `std`.

Implement workloads in order: one request, continuous decode, prefill
interference, mixed service classes, then bursts. Seeds, arrival schedules,
prompt/output token counts, and server settings must be recorded.
