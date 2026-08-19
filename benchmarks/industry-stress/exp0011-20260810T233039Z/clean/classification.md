# Classification: rejected as the canonical correctness baseline

This replay completed 271 requests with zero transport errors, but AIPerf client-side retokenization counted one prescribed 71-token response as 70 and produced an aggregate OSL of 18,914 instead of 18,915.

The HTTP stream contained 71 chunks and `ignore_eos=true`. The corrected `clean_server_tokens/` replay used vLLM usage records and reported the exact 18,915 completion tokens. Use that directory for clean-versus-probe comparisons; retain this directory only as evidence of the client-tokenization boundary.
