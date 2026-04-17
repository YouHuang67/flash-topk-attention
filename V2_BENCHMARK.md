# V2 Benchmark: V1 vs V2 Performance

All benchmarks on a single NVIDIA RTX 4090, bfloat16, `q_block_size = score_block_size`. V1 and V2 use equivalent parameters (`threshold=1.0`, `qblock_topk = q_block_size * K`) so outputs are identical.

## Scoring (score-only, no attention output)

V1 fuses scoring + Bitonic Sort in a single Triton kernel; V2 separates block scoring (CUDA CuTe MMA for D >= 80, Triton otherwise) from top-k selection (CUDA CUB radix sort). The K=32 case shows the largest gap (4.9x) because V1's online Bitonic Sort cost scales with K while V2's block scoring is K-independent.

| B | N | H | D | Block Size | Top-K | V1 | V2 | Speedup |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 2,048 | 32 | 128 | 64 | 16 | 1.18 ms | 0.68 ms | **1.7x** |
| 1 | 4,096 | 32 | 128 | 64 | 16 | 4.27 ms | 2.67 ms | **1.6x** |
| 1 | 8,192 | 32 | 128 | 64 | 16 | 16.59 ms | 10.56 ms | **1.6x** |
| 1 | 4,096 | 32 | 128 | 128 | 16 | 3.57 ms | 2.42 ms | **1.5x** |
| 1 | 4,096 | 32 | 128 | 64 | 32 | 13.11 ms | 2.69 ms | **4.9x** |
| 2 | 4,096 | 32 | 128 | 64 | 16 | 9.31 ms | 5.28 ms | **1.8x** |
| 1 | 2,016 | 32 | 128 | 48 | 16 | 2.01 ms | 0.98 ms | **2.1x** |
| 1 | 4,032 | 32 | 128 | 48 | 16 | 7.14 ms | 3.82 ms | **1.9x** |
| 1 | 3,008 | 32 | 128 | 64 | 16 | 2.83 ms | 1.57 ms | **1.8x** |
| 1 | 5,056 | 32 | 128 | 64 | 16 | 7.82 ms | 4.63 ms | **1.7x** |
| 1 | 4,096 | 4 | 64 | 64 | 16 | 0.62 ms | 0.28 ms | **2.2x** |
| 1 | 4,096 | 32 | 64 | 64 | 16 | 3.99 ms | 2.10 ms | **1.9x** |

## Sparse Attention (given pre-computed merged indices)

V1 sparse attention is a Triton kernel; V2 uses a CUDA CuTe MMA kernel. The 2.3x-4.2x speedup is consistent across pow2 and non-pow2 configurations.

| B | N | H | D | Block Size | Top-K | V1 | V2 | Speedup |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 2,048 | 32 | 128 | 64 | 16 | 4.45 ms | 1.24 ms | **3.6x** |
| 1 | 4,096 | 32 | 128 | 64 | 16 | 17.75 ms | 4.44 ms | **4.0x** |
| 1 | 8,192 | 32 | 128 | 64 | 16 | 70.76 ms | 17.37 ms | **4.1x** |
| 1 | 4,096 | 32 | 128 | 128 | 16 | 16.95 ms | 4.48 ms | **3.8x** |
| 1 | 4,096 | 32 | 128 | 64 | 32 | 19.26 ms | 4.53 ms | **4.2x** |
| 2 | 4,096 | 32 | 128 | 64 | 16 | 35.31 ms | 8.95 ms | **3.9x** |
| 1 | 2,016 | 32 | 128 | 48 | 16 | 7.65 ms | 1.98 ms | **3.9x** |
| 1 | 4,032 | 32 | 128 | 48 | 16 | 30.63 ms | 7.42 ms | **4.1x** |
| 1 | 3,008 | 32 | 128 | 64 | 16 | 9.55 ms | 2.55 ms | **3.7x** |
| 1 | 5,056 | 32 | 128 | 64 | 16 | 26.78 ms | 6.70 ms | **4.0x** |
| 1 | 4,096 | 4 | 64 | 64 | 16 | 0.87 ms | 0.38 ms | **2.3x** |
| 1 | 4,096 | 32 | 64 | 64 | 16 | 5.84 ms | 2.28 ms | **2.6x** |

### Sparse Attention: Varying D (RTX 3090, bfloat16)

B=1, H=8, N=2048, Block Size=64, Top-K=8. V1 Triton has a known BD truncation bug for D > 128, so only V2 vs naive correctness is verified for those cases.

| D | V1 | V2 | Speedup |
|:-:|:-:|:-:|:-:|
| 32 | 0.079 ms | 0.065 ms | **1.22x** |
| 48 | 0.122 ms | 0.140 ms | 0.87x |
| 64 | 0.133 ms | 0.108 ms | **1.24x** |
| 80 | 0.333 ms | 0.203 ms | **1.64x** |
| 96 | 0.335 ms | 0.154 ms | **2.18x** |
| 128 | 0.339 ms | 0.182 ms | **1.86x** |

## End-to-End Pipeline (scoring + merge + attention)

V2 total includes `flash_block_score` + `flash_topk_select` + `flash_qblock_merge` + `flash_sparse_attn`. The merge step (`flash_qblock_merge`) adds < 0.2 ms overhead across all cases.

| B | N | H | D | Block Size | Top-K | V1 Total | V2 Total | Speedup |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 4,096 | 32 | 128 | 64 | 16 | 22.01 ms | 7.26 ms | **3.0x** |
| 1 | 8,192 | 32 | 128 | 64 | 16 | 87.25 ms | 28.15 ms | **3.1x** |
| 1 | 4,096 | 32 | 128 | 64 | 32 | 29.82 ms | 7.37 ms | **4.0x** |
| 2 | 4,096 | 32 | 128 | 64 | 16 | 43.44 ms | 14.42 ms | **3.0x** |
| 1 | 4,032 | 32 | 128 | 48 | 16 | 37.33 ms | 11.25 ms | **3.3x** |
| 1 | 5,056 | 32 | 128 | 64 | 16 | 33.97 ms | 11.42 ms | **3.0x** |
| 1 | 4,096 | 32 | 64 | 64 | 16 | 9.37 ms | 4.47 ms | **2.1x** |

## Sparse Attention Backward

CUDA backward (CUTLASS/CuTe dQ + dKV kernels) vs naive backward (PyTorch autograd on vectorized dense masked attention). RTX 3090, bfloat16, `q_block_size = kv_block_size = 64`. Backward time = total(fwd+bwd) - fwd_only, measured via `triton.testing.do_bench`.

### Varying N (B=1, H=8, D=128, topk=8)

| N | CUDA fwd (ms) | CUDA bwd (ms) | CUDA total (ms) | Naive total (ms) | Speedup |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 256 | 0.048 | 2.09 | 2.14 | 13.15 | **6.1x** |
| 512 | 0.073 | 2.06 | 2.14 | 36.72 | **17.2x** |
| 1,024 | 0.102 | 1.93 | 2.04 | 71.09 | **34.9x** |
| 2,048 | 0.159 | 1.73 | 1.89 | 149.77 | **79.4x** |
| 4,096 (H=4) | 0.159 | 1.96 | 2.12 | 161.02 | **75.9x** |
| 8,192 (H=2) | 0.159 | 1.47 | 1.63 | 194.21 | **118.9x** |

### Varying topk (B=1, H=8, N=2048, D=128)

| topk | CUDA bwd (ms) | Naive total (ms) | Speedup |
|:-:|:-:|:-:|:-:|
| 1 | 1.99 | 27.65 | **13.1x** |
| 2 | 2.01 | 44.77 | **21.2x** |
| 4 | 1.95 | 79.56 | **38.2x** |
| 8 | 1.94 | 149.42 | **71.1x** |
| 16 | 1.91 | 289.24 | **135.2x** |
| 32 | 1.56 | 549.14 | **275.3x** |

### Varying D (B=1, H=8, N=2048, topk=8)

| D | CUDA bwd (ms) | Naive total (ms) | Speedup | dq_err | dk_err | dv_err |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 32 | 2.03 | 146.15 | **69.7x** | 1.0e-04 | 8.7e-05 | 1.2e-03 |
| 48 | 2.97 | 146.35 | **47.2x** | 1.0e-04 | 7.8e-05 | 1.3e-03 |
| 64 | 2.04 | 145.70 | **68.2x** | 1.0e-04 | 7.5e-05 | 1.2e-03 |
| 80 | 2.95 | 151.15 | **48.3x** | 1.0e-04 | 7.1e-05 | 1.2e-03 |
| 96 | 1.20 | 146.28 | **108.2x** | 1.0e-04 | 6.8e-05 | 1.2e-03 |
| 128 | 1.94 | 147.80 | **70.5x** | 1.0e-04 | 6.7e-05 | 1.3e-03 |
| 160 | 1.88 | 146.94 | **70.2x** | 1.0e-04 | 6.5e-05 | 1.2e-03 |
| 256 | 1.57 | 148.47 | **75.9x** | 1.0e-04 | 6.2e-05 | 1.2e-03 |

### Varying dtype (B=1, H=8, N=2048, D=128, topk=8)

| dtype | CUDA bwd (ms) | Naive total (ms) | Speedup |
|:-:|:-:|:-:|:-:|
| bf16 | 1.76 | 148.61 | **77.6x** |
| fp16 | 1.95 | 151.86 | **72.1x** |

### Summary

- **Speedup range**: 6x (N=256) to 275x (topk=32 dense)
- **Median speedup**: 70x
- **Gradient accuracy** (median absolute error vs naive): dQ < 2.4e-4, dK < 9.1e-5, dV < 1.3e-3
