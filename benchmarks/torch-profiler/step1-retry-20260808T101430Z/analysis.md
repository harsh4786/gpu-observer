# Cold versus prefix-cached launch-path experiment

## Controlled input

- Model: `Qwen/Qwen3-0.6B`, BF16, eager execution
- Prompt tokens: 44
- Generated tokens: 1
- Requests: cold, identical warm, identical warm repeat
- Prefix-cache result: cold computed 44; each warm request reused 32 and computed 12

## GPU activity evidence

| Request | GPU kernels | Summed kernel duration |
|---|---:|---:|
| Cold | 378 | 11,141.133 us |
| Warm | 378 | 11,016.357 us |
| Warm repeat | 378 | 11,157.764 us |

Prefix caching did not change the total kernel count in this experiment.

## CUDA launch API evidence

| Request | `cudaLaunchKernel` | `cuLaunchKernel` | `cuLaunchKernelEx` |
|---|---:|---:|---:|
| Cold | 265 | 28 | 85 |
| Warm | 265 | 112 | 1 |
| Warm repeat | 265 | 112 | 1 |

The earlier uprobe observed only `cuLaunchKernel`. The common runtime launches eventually pass
through that driver entry point, while the cold GEMM path moves 84 launches through
`cuLaunchKernelEx`. Therefore the earlier `296 -> 380` observation was a launch-doorway
difference, not an increase in executed GPU kernels.

## Kernel-family substitution

Cold-only transformer-layer GEMMs:

- 56 x `nvjet_sm121_tst_mma_64x32x64_8_32x16x64_tmaAB_alignCD4_bz_TNNN`
- 28 x `nvjet_sm121_tst_mma_96x64x64_4_16x64x64_tmaAB_alignCD4_bz_TNNN`
- 28 x `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8`

Warm-only transformer-layer GEMMs:

- 56 x `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8`
- 56 x `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8`

The 44-token and 12-token matrix shapes select different GEMM implementations, but both paths
execute four GEMM kernels per transformer layer. The host probe must cover both
`cuLaunchKernel` and `cuLaunchKernelEx` before its launch count can be treated as complete.

## Corrected two-entry-point probe validation

The updated Aya probe was attached to both driver entry points and tested live
against the same running EngineCore:

| Request | Total records | `cuLaunchKernel` | `cuLaunchKernelEx` | Malformed | Loss markers |
|---|---:|---:|---:|---:|---:|
| Unique 44-token cold | 381 | 296 | 85 | 0 | 0 |
| 32-token-hit warm | 381 | 380 | 1 | 0 | 0 |

The total is invariant and the 84-record migration between entry points exactly
reconstructs the original discrepancy. The three records outside the
378-kernel model-execution region belong to surrounding request work captured
by the wider live-probe window.
