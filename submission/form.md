# lablab.ai submission form — AI Infra Summit Hackathon

Paste-ready text. Limits are lablab's: title ≤ 50 characters, short
description ≤ 255 characters, long description ≥ 100 words. Counts are checked
by `submission/check-limits.py`.

## Track

TODO: confirm from the form's track dropdown. The published sponsor tracks are
Intel Bimanual VLA, Qualcomm Model-to-device, Intel Physical AI, and SiMa
Physical AI. Closest fit: Qualcomm Model-to-device.

## Title

GPU Observer: see what one LLM request costs

## Short description

Type a prompt and watch vLLM turn it into tokens, scheduler steps, packed GPU rows and real CUDA kernel launches on an NVIDIA DGX Spark, down to the KV-cache blocks each request owns, with every number labeled by how it was measured.

## Long description

Between "request in" and "tokens out", an LLM inference server is opaque. The scheduler batches requests, the model runner packs and reorders their tokens into GPU rows, and CUDA Graphs replay kernels with little host visibility. Profilers show kernels, but not whose work they are.

GPU Observer preserves request identity across that whole chain on an NVIDIA DGX Spark running vLLM V1 and Qwen3-14B: prompt, tokens, scheduler step, authoritative packed rows, actual CUDA kernel intervals from CUPTI, and request-owned KV-cache blocks from Compute Sanitizer device events. A live causal graph builds as you chat, with a per-layer kernel DAG lit by real launches and kernel counts split into thinking and response tokens. Sealed traces can be explored offline, which is what the hosted demo shows.

Every value is labeled measured, reconstructed, matched or unavailable, and each claim states its boundary. For the KV-cache kernel under CUDA Graph replay, per-request block ownership was recovered in single-stream decode with zero mismatches across 7,560 device events. A 15-run overhead study found no configuration distinguishable from unmodified vLLM within a 0.5% resolution.

The observer and experiments were built before the event, from August 19; the hackathon window added the chat-to-token live flow, the opt-in shadow view, and the public viewer.

## Links

- Application URL: https://harsh4786.github.io/gpu-observer/
- Demo platform: GitHub Pages (static viewer; the live system requires a DGX Spark)
- Public repository: https://github.com/harsh4786/gpu-observer
- Video: https://github.com/harsh4786/gpu-observer/raw/main/submission/gpu-observer-walkthrough.mp4 (2:18, 1080p MP4, 7.5 MB; upload this file)
- Slides: https://github.com/harsh4786/gpu-observer/raw/main/submission/deck.pdf (upload this PDF)
- Cover image: https://github.com/harsh4786/gpu-observer/raw/main/submission/cover.png (1920×1080 PNG; upload this file)

## Technologies / tags

vLLM, NVIDIA DGX Spark, CUDA, CUPTI, Compute Sanitizer, LLM inference, GPU observability, Rust, Qwen3
