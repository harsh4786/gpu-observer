# EXP-0006 protocol: Qwen3-14B clean eager baseline

## Purpose

Establish an uninstrumented latency and throughput reference for Qwen3-14B before enabling semantic tracing, CUDA host probes, CUPTI, Nsight Systems, or the PyTorch profiler.

## Server controls

Serve the pinned BF16 model revision with TP=1, eager execution, synchronous scheduling, prefix caching disabled, a 4096-token maximum context, an explicit 8 GiB KV cache, and at most 32 sequences. The explicit KV-cache allocation prevents vLLM from reserving roughly 90% of the DGX Spark's unified memory.

Exact server command:

```text
vllm serve Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --port 8000 --dtype bfloat16 --max-model-len 4096 --kv-cache-memory-bytes 8G --max-num-seqs 32 --enforce-eager --no-async-scheduling --no-enable-prefix-caching
```

The server selected FlashAttention 2. Model loading consumed 27.52 GiB and took 180.397283 seconds. The 8 GiB KV cache holds 52,416 tokens.

## Workload A

Run `vllm bench serve` from a separate CPU-only container over host networking. Use the OpenAI-compatible `/v1/completions` protocol. This `openai` backend label describes the wire protocol; inference is still performed locally by vLLM.

Generate one warm-up request followed by one measured request. Both use exactly 256 random input tokens and 64 output tokens. Set random range ratio to zero, seed to 20260809, temperature to zero, and ignore EOS so output length is fixed. Allow only one request at a time.

Preserve detailed per-request TTFT, inter-token intervals, generated text, errors, and aggregate metrics in `workload-a.json`.

## Acceptance checks

- One measured request completed and none failed.
- Total input and output counts are exactly 256 and 64.
- The warm-up request is excluded from measured totals.
- The raw 63 inter-token intervals are retained.
- No instrumentation is active.

Do not interpret request-level percentiles from this workload: there is only one measured request. Workload A verifies correctness and provides a single-request operating point. Workload B will measure a distribution and saturated decode behavior.
