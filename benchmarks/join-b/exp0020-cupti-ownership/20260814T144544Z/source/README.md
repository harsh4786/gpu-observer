# EXP-0020 source snapshot

This directory preserves the source needed to audit the offline join and the exact CUPTI agent used by the measured run.

- `agent/cupti_activity_agent.cpp`: reconstructed measured source, before post-run disable-then-flush hardening.
- `agent/Makefile`: exact build recipe.
- `agent/build-measured/libgpu_observer_cupti_activity.so`: rebuilt against CUDA 13.2.1.
- `collector/join_b_cupti_cache.rs`: offline correlator source.
- `harness/run-join-b-cupti-ownership.sh`: successful-run harness.

The rebuilt measured agent SHA-256 is:

```text
1f8d59eb28d01f176acb1847e43d912fcda58d6aa2868bdc418d5e37fe10ecd9
```

It exactly matches the agent hash recorded in `manifest.txt`. The repository's current agent source is intentionally newer: it disables CUPTI activity kinds before flushing so process shutdown cannot extend a completed window.
