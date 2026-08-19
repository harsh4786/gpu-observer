# Qwen3-14B CUPTI engine-step analysis

## Clock normalization

- Session origin in CLOCK_MONOTONIC: `403339435433289` ns
- Before/after offset drift: `40` ns
- Clock-pair uncertainty: `800` / `976` ns

## Per-step device execution

| Step | Phase | Wall ms | Kernels | Kernel sum ms | GPU busy union ms | First GPU lag ms | Last GPU to end ms | Internal GPU gaps ms | Event-sync wait ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | prefill | 135.176840 | 538 | 133.418144 | 133.397312 | 0.452312 | 0.046544 | 1.280672 | 121.410528 |
| 5 | decode | 122.548561 | 538 | 121.338496 | 121.338496 | 0.130673 | 0.045984 | 1.033408 | 112.219296 |
| 6 | decode | 123.327695 | 538 | 122.158976 | 122.158976 | 0.148416 | 0.043823 | 0.976480 | 113.584688 |

## Integrity

- Semantic steps: `3`
- CUPTI kernels: `1614`
- Unassigned kernels: `0`
- Kernels without runtime correlation: `0`
- CUDA streams: `1`
- Summed kernel time: `376.915616` ms
- Overlap-safe GPU busy time: `376.894784` ms
- cudaEventSynchronize host wait: `347.214512` ms
- Memcpy activities: `113` totaling `823459` bytes
- Memset activities: `123` totaling `87052` bytes

## Top kernels by device time

| Kernel | Calls | Device ms |
|---|---:|---:|
| `kernel` | 323 | 240.231200 |
| `nvjet_sm121_tst_mma_192x128x64_2_96x32x64_tmaAB_alignCD4_bz_TNNN` | 40 | 64.063840 |
| `Kernel2` | 120 | 54.601888 |
| `unrolled_elementwise_kernel` | 144 | 13.823360 |
| `flash_fwd_splitkv_kernel` | 120 | 0.801696 |
| `fused_add_rms_norm_kernel` | 240 | 0.796800 |
| `rotary_embedding_kernel` | 120 | 0.665728 |
| `reshape_and_cache_flash_kernel` | 120 | 0.660256 |
| `act_and_mul_kernel` | 120 | 0.605408 |
| `rms_norm_kernel` | 243 | 0.553760 |
| `reduce_kernel` | 3 | 0.052704 |
| `index_elementwise_kernel` | 6 | 0.027968 |

## Interpretation boundary

**Measured:** CUPTI device intervals nearly fill each semantic engine step, and the host blocks in one `cudaEventSynchronize` call per step while that queued work executes.

**Inferred:** The earlier Aya-to-step-end gap is predominantly asynchronous GPU execution being drained behind the event synchronization, not an idle CPU-only delay.

**Unknown:** This instrumented run does not establish uninstrumented latency, memory-bandwidth saturation, unified-memory migration, or per-request work inside a mixed-batch kernel.
