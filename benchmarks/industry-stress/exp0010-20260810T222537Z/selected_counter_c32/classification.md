# Classification: invalid as a device-counter overhead arm

This arm completed 100 requests, but it is not evidence for block-counter overhead.

- `device-probe.summary.tsv` reports `modules_patched=0`.
- The allowlisted kernel did not execute under this 1,024-input/128-output, concurrency-32 workload.
- The generated `inputs.json` differs from the clean concurrency-32 input file despite the same CLI seed.

The measured performance therefore represents subscriber-loaded/no-module-patched behavior on a different prompt corpus. It is retained as raw evidence and excluded from the causal clean-versus-counter comparison.

