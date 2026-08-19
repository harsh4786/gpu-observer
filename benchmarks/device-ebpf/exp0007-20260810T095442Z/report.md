# EXP-0007 — bpftime and gpu_ext compatibility baselines on GB10

Date: 2026-08-10 UTC

## Outcome

bpftime executed an injected eBPF probe inside a standalone CUDA kernel on the
DGX Spark GB10. gpu_ext's ARM64 policy toolchain compiled successfully, but its
runtime cannot attach to the currently loaded NVIDIA 580.173.02 driver because
the live module does not contain gpu_ext's custom hooks or BTF `struct_ops`
types.

This is a compatibility result, not an overhead or vLLM-compatibility result.
No NVIDIA kernel module was unloaded, loaded, installed, or replaced.

## Observer boundaries

```text
CUDA application correctness
  observed by: synchronized vector-add result on the host

PTX interception and replacement
  observed by: bpftime agent logs

in-kernel probe execution
  observed by: one device event carrying GPU globaltimer + block/thread index

gpu_ext build compatibility
  observed by: clang/bpftool/host-linker outputs

gpu_ext live attach compatibility
  observed by: live kallsyms and live-kernel BTF inventory
```

No observer in this experiment measures kernel duration or inference latency.

## bpftime

### Control

The finite vector-add workload launched 32 blocks of 128 threads, synchronized,
and returned the expected values:

```text
vectorAdd C[0]=0.0 C[1]=4.0 ... status=PASS
```

### Compatibility failures found

1. bpftime rewrote the PTX header to PTX 8.7 for `sm_121`; CUDA 13 ptxas
   rejected that pair. The local fix selects PTX 9.0 for `sm_121`.
2. CUDA 13 passes a CUDA kernel handle to `cudaLaunchKernel`. bpftime treated
   it as a host function address and resolved `_fini`, so it never selected the
   patched kernel. The local fix resolves CUDA 13 handles with
   `cuKernelGetName`, then falls back to the old host-symbol path.

The full dirty worktree is preserved in `bpftime-worktree-full.patch`; the two
files changed during this experiment are preserved in
`bpftime-exp0007-changes.patch`. Other bpftime modifications predated EXP-0007
and are not attributed to this experiment.

### Measured success

- target process exit: `0`
- synchronized CUDA result: correct
- ptxas target: `sm_121`
- late-launch mapping: original kernel name to patched `CUfunction`
- emitted events: exactly `1`
- event payload: `ts=448699029972160 block=(0,0,0) thread=(0,0,0)`
- bpftime attach regression suite: 94 assertions in 21 test cases passed
- tracer exit: `124`, expected because a bounded timeout stopped the collector
  after the finite target exited

Exactly one event is expected: the probe filters for block zero, thread zero.
The patched kernel used 32 registers and ptxas reported a 16,464-byte cumulative
stack size. That is an overhead warning, not an overhead measurement.

## gpu_ext

### Source and build gate

- repository release: `v0.0.21`
- commit: `2cf51085670c3d64bdd7a5ec6b612570a9fd9440`
- modified NVIDIA module base: `575.57.08`
- live NVIDIA driver: `580.173.02`
- host: AArch64, Linux `6.17.0-1029-nvidia`
- live kernel BTF was used to generate the ARM64 `vmlinux.h`

Two policies compiled successfully:

1. `chunk_trace`: eBPF object, CO-RE skeleton, and AArch64 loader
2. `struct_ops`: eBPF object with `.struct_ops`, CO-RE skeleton, and AArch64
   loader

The first build attempt failed because the nested pinned `bpftool/libbpf`
submodule was not initialized. That failure and the successful rerun are both
preserved.

### Runtime gate

The live 580 stack contains none of the required custom interfaces:

- required gpu_ext BTF types found: `0/2` (`gpu_mem_ops`, `gpu_sched_ops`)
- required `chunk_trace` hook symbols found: `0/3`
- NVIDIA/UVM module BTF files exposed in `/sys/kernel/btf`: none

There is also source drift inside the pinned release: `chunk_trace.bpf.c`
requests older `uvm_bpf_call_pmm_*` hook names that do not appear even in the
pinned modified module source, whose current functions are named
`uvm_bpf_call_gpu_*`. Therefore `chunk_trace` is not a trustworthy runtime
smoke target without first repairing its hook names.

Running a gpu_ext policy now would not test GB10 behavior; it would only prove
that stock NVIDIA 580 lacks gpu_ext's custom ABI. Loading the supplied 575-based
module into a live 580 userspace stack was deliberately rejected as unsafe.

## Claims

### Measured

- bpftime can inject, execute, and transport one selected device-side event on
  this GB10/CUDA 13 stack after the two documented fixes.
- gpu_ext policies can compile into valid ARM64 host and eBPF artifacts here.
- the currently loaded driver lacks gpu_ext's required runtime interfaces.

### Inferred

- bpftime is the shorter path to a device-side observability demo because it
  works without replacing the NVIDIA kernel module.
- gpu_ext requires a deliberate 575-to-580 driver-hook port before runtime
  experiments on this machine.

### Still unknown

- bpftime overhead at different probe densities
- bpftime behavior on Triton JIT, marlin, or any vLLM kernel
- whether the gpu_ext driver hooks can be ported safely to NVIDIA 580/GB10
- whether either mechanism preserves CUDA Graph behavior

## Next discriminating experiment

Run the same finite kernel for many iterations under CUPTI with four probe
densities:

```text
no probe
  -> one event per kernel
  -> one event per block
  -> one event per thread
```

Record overlap-safe kernel duration, event loss, register count, stack size,
and output correctness. This separates “it executes” from “it is usable.” Only
after that passes should bpftime be tried on a Triton JIT kernel and then a
selected vLLM kernel.

For gpu_ext, the next experiment is source-only: diff the 575 hook sites against
the installed/open 580 source and compile the port. Do not load it until there
is an isolated recovery and reboot plan.
