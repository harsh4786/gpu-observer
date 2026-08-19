# EXP-0010 adaptive branch: corrected selected-kernel arm

Created after inspecting the first selected-counter arm and before launching the corrected server.

## Evidence forcing the correction

The first arm reported `modules_patched=0`: its allowlisted kernel never ran. AIPerf also produced a different prompt corpus than the clean concurrency-32 arm despite the same random seed. That run cannot measure device-counter overhead.

## Corrected design

- Replay the clean concurrency-32 `inputs.json` through AIPerf's `inputs-json` loader, which sends saved payloads verbatim.
- Target `nvjet_sm121_tst_mma_192x144x64_2_48x72x64_tmaAB_alignCD4_bz_TNNN`, the most frequently launched SM121 kernel in the first arm (`6,640` launches).
- Keep model, server configuration, concurrency, warmup count, request count, token lengths, and endpoint unchanged.
- Accept the run only if there are 100 valid 128-token responses, zero errors, no fatal CUDA messages, `modules_patched >= 1`, and a nonzero device callback count.

