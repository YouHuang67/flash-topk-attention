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
