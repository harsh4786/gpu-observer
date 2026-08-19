# EXP-0002 claims

## Supported

- The external uprobe path captured 1,143 fixed-size CUDA launch records during the request.
- The patched synchronous `EngineCore.step` boundary emitted zero records on the default NGC 26.05 configuration.
- The installed runtime auto-enabled asynchronous scheduling and selected the queued step path.
- Container and host PIDs are different identifiers and require an explicit namespace mapping.

## Falsified

- TP=1 is sufficient to conclude that `EngineCore.step` is the active vLLM V1 semantic boundary.
- A host user can always map the root-created semantic ring directly when the server runs as root in Docker.

## Not established

- GPU kernel durations; launch submission timestamps are not execution durations.
- Whether the async queued path can be represented by the same begin/end semantics without carrying the semantic step ID in the queue tuple.
