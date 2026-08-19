# EXP-0009 adaptive branch after block-counter failure

Created before the subscriber-only and block-no-op requests.

## Triggering evidence

The clean arm completed an identical 128-input/16-output request. The
block-counter arm produced one token and then EngineCore reported
`CUDA error: misaligned address` at the asynchronous result synchronization
boundary. The AIPerf input files have identical SHA-256 hashes.

AIPerf recorded zero request errors despite the truncated one-token stream, and
the container ultimately reported exit status zero. Therefore HTTP status and
container exit status are insufficient correctness gates; exact output length
and server fatal-error scans are mandatory.

## Added diagnostic arms

1. `subscriber`: the Compute Sanitizer subscriber is loaded but no SASS patch
   is applied.
2. `block_noop`: a block-entry callback is inserted and immediately returns,
   without reading callback data or modifying device state.

The same model configuration and exact AIPerf request are retained.

## Interpretation matrix

- subscriber fails: host subscriber/injection is graph-incompatible;
- subscriber passes, no-op fails: inserted callback/control-flow is unsafe;
- no-op passes, counter fails: callback-data pointer, device allocation, or
  atomic counter path is unsafe under graph replay;
- all pass on rerun: investigate nondeterministic state and refuse the long
  benchmark until the fault is reproducible and explained.
