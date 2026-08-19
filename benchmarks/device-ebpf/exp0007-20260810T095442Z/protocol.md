# EXP-0007 — GPU eBPF compatibility baselines

Date: 2026-08-10 UTC

## Causal questions

1. Can the locally patched bpftime runtime inject and execute an eBPF probe
   inside a standalone CUDA kernel on the DGX Spark GB10 (`sm_121`)?
2. Is a buildable gpu_ext implementation available, and does this host satisfy
   the prerequisites for a non-destructive build experiment?

These are compatibility smoke tests. They do not measure production overhead
and do not establish compatibility with vLLM kernels.

## Observation points

| Observer | Location | Exact event | Timestamp | Cannot establish |
|---|---|---|---|---|
| CUDA sample stdout | Host application | Correct vector-add result after synchronization | Host receipt; no recorded clock | Whether the eBPF probe executed |
| bpftime agent log | CUDA userspace loading path | Fatbinary interception, PTX rewrite, and kernel replacement | Log order only | Actual GPU execution duration |
| bpftime device probe | Injected GPU code | Selected kernel entry plus block/thread coordinates | GPU `globaltimer`, nanoseconds | End-to-end request latency |
| `nvidia-smi` | Driver/host | GPU identity, load, temperature, active contexts | Sampling time | Per-kernel behavior |
| gpu_ext build gate | Host compiler and source tree | ARM64 BPF object, CO-RE skeleton, and loader generation | Build completion only | Whether the live driver exposes the required hooks |
| live-kernel symbol/BTF gate | Running kernel and NVIDIA modules | Required hook symbols and `struct_ops` types | Not a timing observer | Whether a future port would behave correctly |

## Event sequence under test

```text
sample registers CUDA fatbinary
  -> bpftime agent intercepts registration
  -> eBPF bytecode is translated/injected into PTX
  -> vectorAdd launches
  -> injected device probe executes
  -> probe event reaches the host collector
  -> vectorAdd result remains correct
```

## Pass and rejection conditions

### bpftime

Pass only if all are observed:

- the uninstrumented vector-add result is correct;
- the instrumented result remains correct;
- the loader reports successful PTX/kernel instrumentation;
- at least one event contains valid block/thread coordinates and a non-zero
  GPU global-timer value;
- neither process reports a CUDA or bpftime fatal error.

Reject or mark partial if the result is correct but there is no device event;
that proves the CUDA workload, not GPU eBPF execution.

### gpu_ext

The source/build gate passes only if a versioned source artifact and its exact
NVIDIA open-driver/kernel prerequisites can be identified and a minimal ARM64
policy produces BPF bytecode, a CO-RE skeleton, and a user-space loader. Runtime
attachment passes only if the live NVIDIA module exposes the required hooks
and BTF types. The running driver must not be replaced or reloaded during this
experiment.

## Controlled variables

- Hardware: NVIDIA GB10, single GPU
- Host CUDA toolkit: 13.0
- bpftime commit and local diff: recorded with the run
- Workload: standalone vector addition
- vLLM: not running and not part of this test

## Follow-up, not part of this smoke test

If bpftime passes, separately benchmark no probe versus kernel-entry,
one-thread-per-block, and per-thread probe densities with CUPTI timing. A
successful smoke test must not be reported as low overhead.
