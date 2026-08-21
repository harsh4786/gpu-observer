# EXP-0024 claim boundary

## Supported

For this pinned Qwen3-14B mixed workload on GB10, capping individual prefill slices at 256 tokens only while interactive decodes were active reduced interactive p99 TTFT by 18.227% (95% paired-t CI 14.403–22.051% reduction), reduced p99 ITL by 2.767%, increased output throughput by 1.778%, and increased the background completion window by only 0.239%. Every one of five paired runs moved p99 TTFT in the favorable direction, every request completed, and every condition-matching mixed step obeyed the cap.

## Not supported

- Mean TTFT did not improve conclusively; its interval crossed zero.
- The result does not prove 256 is the optimal cap or that the policy generalizes to other models, hardware, loads, or prompt distributions.
- The policy does not yet use an explicit service-class label and is not a production admission-control implementation.
- More mixed steps are not automatically worse; the controlled arm intentionally created more, cheaper mixed steps.
- Five pairs do not yield a strong distribution-free significance claim.
- The run does not measure CUPTI/Sanitizer overhead or arbitrary per-request instruction cost.
