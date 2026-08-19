# Qwen3-14B manual reproduction

This reproduction correlates one deterministic inference request with the vLLM engine steps
that serve it and the CUDA launches submitted during those steps. The reliable path is:

`request -> semantic request slice -> engine step time interval -> CUDA launches`

Load the fixed inputs in every terminal:

```bash
source /home/harsh4786/gpu-observer/repro/qwen3-14b/env.sh
source /home/harsh4786/gpu-observer/repro/qwen3-14b/active-run.env
```

Create the configured output directory once:

```bash
mkdir -p "$GO_RUN/semantic-shm"
printf '%s\n' "$GO_RUN"
```

The conversational reproduction guide contains the ordered commands and explains their
meaning. This file deliberately holds only stable inputs and the deterministic request.
