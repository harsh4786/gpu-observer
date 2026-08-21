# Rejected harness pilot

This EXP-0024 pilot completed its first control workload but is not performance evidence. The post-run SSE validator used an invalid regular expression for the literal `[DONE]` marker and exited before the arm could pass all predeclared integrity gates. The raw first-arm files are preserved for debugging only. The validator was replaced by exact fixed-string, whole-line matching and the campaign was restarted from pair 1.
