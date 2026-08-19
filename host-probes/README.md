# Host probes

External CUDA runtime/driver attachment lives here. Launch coverage starts with
both `cuLaunchKernel` and `cuLaunchKernelEx`, then expands to copies,
synchronization, graph launch, and allocation.

Each attached process owns a bounded SPSC producer. Probes emit numeric compact
records only; symbolization, JSON, and file I/O belong in the collector. A full
ring drops visibly and never stalls the CUDA submission thread.

The first implementation must compare launch count, stream, geometry, and
timestamps with Nsight Systems before expanding coverage.

## CUDA launch probes

The first real probes are split into three crates:

- `common`: `no_std`, integer-only, fixed 88-byte wire ABI;
- `ebpf`: `no_std` Aya uprobe and bounded 8 MiB kernel ring;
- `loader`: cold-path Aya loader and ring drain.

The wire record contains raw `bpf_ktime_get_ns`, PID, TID, CPU-local sequence,
function pointer, CUDA stream, grid/block dimensions, and dynamic shared-memory
bytes. A sequence gap or `DROPPED_BEFORE` bit makes loss visible. The producer
uses `BPF_RB_NO_WAKEUP`; the loader drains every 10 ms, avoiding a wakeup on
every CUDA launch.

The record deliberately has no CUPTI correlation ID: that value is internal to
CUPTI and is not an argument of `cuLaunchKernel`. Exact submission-to-GPU
execution correlation must use CUPTI API activity -> correlation ID -> CUPTI
kernel activity. Do not synthesize false precision by treating uprobe order as
that identifier.

Aya's kernel `RingBuf` is shared across CPUs; it is a bounded staging transport,
not the collector's SPSC ring. TP=1 currently shows one CUDA-launch TID, so this
is a low-contention first implementation. If later traces show concurrent
launch TIDs, benchmark it against per-CPU perf buffers before trusting overhead
results.

On AArch64, `CUstream` is the ninth `cuLaunchKernel` argument. The first eight
arguments occupy `x0..x7`, so the probe reads the stream from the first stack
slot with `bpf_probe_read_user`. The eBPF crate deliberately sets
`bpf_target_arch=aarch64` in its build script; this first implementation is
DGX Spark-specific and must not silently compile the generic register path.

`cuLaunchKernelEx` passes a pointer to `CUlaunchConfig` followed by the kernel
function. Its probe makes one bounded user-memory read of the 56-byte AArch64
CUDA 13 layout and marks the record with `EXTENDED_CONFIG`. The fixed wire
record remains 88 bytes.

This second entry point is required for correctness, not optional enrichment.
On Qwen3-0.6B, a 44-token cold prefill selected NVJet GEMMs and issued 85
`cuLaunchKernelEx` calls, while the 12-token prefix-cached path selected small
CUTLASS WMMA GEMMs and issued only one. Probing only `cuLaunchKernel` made two
equal 378-kernel executions appear to contain 296 and 380 launches.

Build the normal loader:

```bash
cargo build --release -p gpu-observer-host-probe
```

Build the eBPF object after installing nightly Rust and `bpf-linker`:

```bash
cargo +nightly build \
  --manifest-path host-probes/ebpf/Cargo.toml \
  --release --target bpfel-unknown-none -Z build-std=core
```

Run as root, targeting the library inode visible through the process root:

```bash
sudo target/release/gpu-observer-host-probe \
  193408 \
  /proc/193408/root/usr/local/cuda-13.2/compat/lib.real/libcuda.so.595.58.03 \
  host-probes/ebpf/target/bpfel-unknown-none/release/gpu-observer-host-probe \
  15
```

The PID and exact library path are run-specific; resolve them again after each
vLLM restart.
