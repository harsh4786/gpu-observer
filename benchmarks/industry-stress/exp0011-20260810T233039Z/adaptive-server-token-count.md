# EXP-0011 adaptive branch: server-reported token counts

Created after the first clean replay and before the corrected clean server was launched.

## Triggering evidence

The first clean replay completed 271 requests with no transport errors, but AIPerf client-side retokenization reported 18,914 output tokens versus 18,915 prescribed. The only mismatch was a 387-input/71-output request counted as 70. Its HTTP stream contained 71 chunks and the request included `ignore_eos=true`.

This is consistent with decode-then-reencode non-round-tripping for one generated string, not necessarily an engine correctness failure. AIPerf's `osl_mismatch_count` nevertheless reported zero, so that field is not a sufficient gate.

## Corrected design

Repeat both accepted arms with `--use-server-token-count`. For OpenAI-compatible streaming chat, AIPerf automatically requests usage records via `stream_options.include_usage=true`, so vLLM's generated-token count becomes authoritative. Retain the first replay under `clean/` as a rejected client-tokenization diagnostic; write the corrected baseline under `clean_server_tokens/`.

All trace, scheduling, server, correctness, and thermal controls remain unchanged.

## Preflight gate

Before the second five-minute replay, send only the isolated 387-input/71-output trace row. Proceed only if server-reported ISL and OSL are exactly 387 and 71 with zero errors.

## Preflight result and semantic correction

The server reported OSL 71 exactly and zero errors. It reported 395 prompt tokens, not 387, because vLLM counts the eight-token chat-template wrapper that AIPerf adds around the prescribed user-text length.

The accepted full-run gates are therefore 18,915 server-reported completion tokens and 144,027 server-reported prompt tokens (141,859 trace tokens plus 8 x 271 chat-template tokens). Keep trace-level ISL and server-billed prompt tokens as separate metrics.
