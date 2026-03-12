English | [中文](BENCHMARK_CN.md)

# Benchmark

### Baselines

| Method | Description |
|--------|-------------|
| **Naive** | Standard PyTorch attention + `torch.topk` (two separate passes) |
| **Flash Attention 2** | Forward pass only — computes attention output but **no TopK scoring** (lower bound reference) |
| **Flash Scoring** | This work — computes full attention output **and** top-k block indices in a single kernel pass |

---

## Summary by Sequence Length Scale

| Scale | Sequence Length | Configurations | Average vs Naive (speedup) | Average vs Flash Attention 2 (overhead) |
|-------|:-:|:-:|:-:|:-:|
| Small | 2048 – 4096 | 12 | **4.48×** faster | 1.98× |
| Medium | 4096 – 8192 | 25 | **5.21×** faster | 1.82× |
| Large | 8192 – 16384 | 25 | **4.80×** faster | 2.29× |
| Extra Large | 16384 – 32768 | 14 | — | 3.20× |
| Ultra Large | 32768 – 470016 | 33 | — | 2.88× |

> Naive baseline ran out of GPU memory (OOM) for sequences beyond 16384 and could not be executed (SKIPPED in raw results).

---

## Overhead vs Top-K

| Top-K | Configurations | Average overhead vs Flash Attention 2 |
|:-----:|:-:|:-:|
| 1 | 3 | 1.42× |
| 2 | 3 | 1.30× |
| 4 | 13 | 2.10× |
| 8 | 21 | 1.83× |
| 16 | 27 | 1.73× |
| 32 | 25 | 2.35× |
| 64 | 21 | 5.19× |

Overhead grows with Top-K due to Bitonic Sort pool maintenance. For practical values (Top-K ≤ 32), overhead stays within **1.3–2.5×** of Flash Attention 2.

---

## Power-of-2 vs Non-Power-of-2 Block Sizes

| Block Size Type | Configurations | Average overhead vs Flash Attention 2 |
|:----------------|:-:|:-:|
| Power-of-2 | 81 | 2.28× |
| Non-power-of-2 | 32 | 3.24× |

Non-power-of-2 block sizes incur extra overhead from padding and boundary handling in the kernel.

---

## Full Results

> SKIPPED: Naive baseline ran out of GPU memory (OOM) and could not be executed.

```
Scale        | Batch | Seq Len | Heads | Block Size | Top-K | Naive (ms) |  FA2 (ms) | Scoring (ms) | Scoring / FA2
----------------------------------------------------------------------------------------------------------------------
Extra Small  |     1 |     512 |     8 |         16 |     4 |      0.332 |     0.045 |        0.272 |         6.05x
Extra Small  |     1 |     512 |     8 |         32 |     4 |      0.322 |     0.049 |        0.279 |         5.67x
Extra Small  |     1 |    1024 |     8 |         32 |     8 |      0.452 |     0.057 |        0.269 |         4.74x
Extra Small  |     1 |    1024 |     8 |         64 |     8 |      0.412 |     0.050 |        0.274 |         5.45x
Small        |     1 |    1920 |     8 |        48* |     8 |      1.322 |     0.121 |        0.271 |         2.25x
Small        |     1 |    1920 |     8 |        60* |     8 |      1.258 |     0.122 |        0.273 |         2.24x
Small        |     1 |    1920 |     8 |        96* |     4 |      1.203 |     0.122 |        0.306 |         2.51x
Small        |     1 |    2048 |     8 |         64 |    16 |      1.370 |     0.170 |        0.305 |         1.80x
Small        |     1 |    2048 |     8 |        128 |     8 |      1.323 |     0.169 |        0.306 |         1.82x
Small        |     1 |    4096 |     4 |         64 |     1 |      2.538 |     0.331 |        0.464 |         1.40x
Small        |     1 |    4096 |     4 |         64 |     2 |      2.566 |     0.395 |        0.449 |         1.13x
Small        |     1 |    4096 |     4 |         64 |     4 |      2.664 |     0.394 |        0.445 |         1.13x
Small        |     1 |    4096 |     4 |         64 |     8 |      2.627 |     0.394 |        0.493 |         1.25x
Small        |     1 |    4096 |     4 |         64 |    16 |      2.588 |     0.392 |        0.586 |         1.49x
Small        |     1 |    4096 |     4 |         64 |    32 |      2.735 |     0.394 |        0.909 |         2.31x
Small        |     1 |    4096 |     4 |         64 |    64 |      2.727 |     0.391 |        1.736 |         4.44x
Medium       |     1 |    3840 |     8 |        48* |    16 |      4.589 |     0.450 |        1.220 |         2.71x
Medium       |     1 |    3840 |     8 |        60* |    16 |      4.614 |     0.526 |        1.000 |         1.90x
Medium       |     1 |    3840 |     8 |        96* |     8 |      4.525 |     0.543 |        0.811 |         1.49x
Medium       |     1 |    3840 |     8 |       120* |     8 |      4.472 |     0.543 |        0.753 |         1.39x
Medium       |     1 |    3840 |     8 |       192* |     4 |      4.271 |     0.545 |        0.680 |         1.25x
Medium       |     1 |    4096 |    16 |         64 |    16 |     11.052 |     1.009 |        2.108 |         2.09x
Medium       |     1 |    4096 |     8 |         64 |    16 |      5.649 |     0.555 |        1.089 |         1.96x
Medium       |     2 |    4096 |     8 |         64 |    16 |     10.961 |     1.011 |        2.074 |         2.05x
Medium       |     1 |    4096 |     8 |        128 |    16 |      5.250 |     0.646 |        0.942 |         1.46x
Medium       |     1 |    4096 |     8 |        256 |     8 |      5.222 |     0.642 |        0.756 |         1.18x
Medium       |     1 |    7680 |     4 |        48* |    32 |      9.850 |     0.913 |        2.994 |         3.28x
Medium       |     1 |    7680 |     4 |        60* |    32 |      9.458 |     1.047 |        2.630 |         2.51x
Medium       |     1 |    7680 |     4 |        96* |    16 |      8.682 |     1.040 |        2.051 |         1.97x
Medium       |     1 |    7680 |     4 |       120* |    16 |      8.674 |     1.055 |        1.683 |         1.60x
Medium       |     1 |    7680 |     4 |       192* |     8 |      8.430 |     1.037 |        1.341 |         1.29x
Medium       |     1 |    7680 |     4 |       384* |     4 |      8.360 |     1.059 |        1.408 |         1.33x
Medium       |     1 |    8192 |     4 |        128 |     1 |     10.303 |     1.123 |        1.609 |         1.43x
Medium       |     1 |    8192 |     4 |        128 |     2 |     10.370 |     1.268 |        1.668 |         1.31x
Medium       |     1 |    8192 |     4 |        128 |     4 |     10.434 |     1.274 |        1.760 |         1.38x
Medium       |     1 |    8192 |     4 |        128 |     8 |     10.458 |     1.255 |        1.742 |         1.39x
Medium       |     1 |    8192 |     4 |        128 |    16 |     10.444 |     1.292 |        1.963 |         1.52x
Medium       |     1 |    8192 |     4 |        128 |    32 |     10.451 |     1.262 |        3.026 |         2.40x
Medium       |     1 |    8192 |     4 |        128 |    64 |     10.772 |     1.285 |        5.055 |         3.93x
Medium       |     1 |    8192 |     4 |        512 |     8 |     10.279 |     1.277 |        1.737 |         1.36x
Medium       |     1 |    8192 |     4 |       1024 |     4 |     10.254 |     1.283 |        1.705 |         1.33x
Large        |     4 |    4096 |     8 |         64 |    16 |     21.471 |     2.106 |        4.138 |         1.96x
Large        |     1 |    8192 |     8 |         64 |    32 |     21.833 |     2.191 |        6.000 |         2.74x
Large        |     2 |    8192 |     4 |        128 |    16 |     19.872 |     2.219 |        3.824 |         1.72x
Large        |     1 |    8192 |    16 |        128 |    16 |     39.693 |     4.316 |        7.431 |         1.72x
Large        |     1 |    8192 |     8 |        128 |    16 |     20.259 |     2.284 |        3.759 |         1.65x
Large        |     1 |    8192 |     8 |        128 |    32 |     20.233 |     2.303 |        5.636 |         2.45x
Large        |     1 |    8192 |     8 |        256 |    16 |     19.934 |     2.286 |        3.797 |         1.66x
Large        |     1 |   15360 |     4 |        48* |    64 |    SKIPPED |     4.195 |       26.733 |         6.37x
Large        |     1 |   15360 |     4 |        60* |    64 |    SKIPPED |     3.892 |       24.499 |         6.30x
Large        |     1 |   15360 |     4 |        96* |    32 |    SKIPPED |     3.892 |       11.158 |         2.87x
Large        |     1 |   15360 |     4 |       120* |    32 |    SKIPPED |     4.219 |       10.203 |         2.42x
Large        |     1 |   15360 |     4 |       192* |    16 |    SKIPPED |     4.276 |        7.176 |         1.68x
Large        |     1 |   15360 |     4 |       384* |     8 |    SKIPPED |     4.342 |        5.953 |         1.37x
Large        |     1 |   16384 |     4 |         64 |    32 |    SKIPPED |     4.525 |       12.470 |         2.76x
Large        |     1 |   16384 |     4 |        128 |    32 |    SKIPPED |     4.954 |       11.487 |         2.32x
Large        |     1 |   16384 |     4 |        256 |     1 |    SKIPPED |     4.817 |        6.878 |         1.43x
Large        |     1 |   16384 |     4 |        256 |     2 |    SKIPPED |     4.863 |        7.119 |         1.46x
Large        |     1 |   16384 |     4 |        256 |     4 |    SKIPPED |     4.862 |        6.656 |         1.37x
Large        |     1 |   16384 |     4 |        256 |     8 |    SKIPPED |     4.832 |        6.454 |         1.34x
Large        |     1 |   16384 |     4 |        256 |    16 |    SKIPPED |     4.816 |        7.620 |         1.58x
Large        |     1 |   16384 |     4 |        256 |    32 |    SKIPPED |     4.771 |       10.728 |         2.25x
Large        |     1 |   16384 |     4 |        256 |    64 |    SKIPPED |     4.730 |       16.218 |         3.43x
Large        |     1 |   16384 |     4 |        512 |    16 |    SKIPPED |     4.602 |        7.122 |         1.55x
Large        |     1 |   16384 |     4 |       1024 |     8 |    SKIPPED |     4.729 |        6.857 |         1.45x
Large        |     1 |   16384 |     4 |       2048 |     4 |    SKIPPED |     4.907 |        6.393 |         1.30x
Extra Large  |     2 |   16384 |     4 |        256 |    16 |    SKIPPED |     9.142 |       15.229 |         1.67x
Extra Large  |     1 |   30720 |     4 |        48* |    64 |    SKIPPED |    16.470 |      108.990 |         6.62x
Extra Large  |     1 |   30720 |     4 |        60* |    64 |    SKIPPED |    16.352 |      101.345 |         6.20x
Extra Large  |     1 |   30720 |     4 |        96* |    64 |    SKIPPED |    16.946 |       93.411 |         5.51x
Extra Large  |     1 |   30720 |     4 |       120* |    64 |    SKIPPED |    17.273 |       74.687 |         4.32x
Extra Large  |     1 |   30720 |     4 |       192* |    32 |    SKIPPED |    17.443 |       39.834 |         2.28x
Extra Large  |     1 |   30720 |     4 |       384* |    16 |    SKIPPED |    17.790 |       26.780 |         1.51x
Extra Large  |     1 |   32768 |     4 |         64 |    64 |    SKIPPED |    18.524 |      112.299 |         6.06x
Extra Large  |     1 |   32768 |     4 |        128 |    32 |    SKIPPED |    18.870 |       46.868 |         2.48x
Extra Large  |     1 |   32768 |     4 |        256 |    32 |    SKIPPED |    19.812 |       44.548 |         2.25x
Extra Large  |     1 |   32768 |     4 |        512 |    16 |    SKIPPED |    19.613 |       29.730 |         1.52x
Extra Large  |     1 |   32768 |     4 |       1024 |    16 |    SKIPPED |    18.874 |       29.142 |         1.54x
Extra Large  |     1 |   32768 |     4 |       2048 |     8 |    SKIPPED |    18.700 |       26.949 |         1.44x
Extra Large  |     1 |   32768 |     4 |       4096 |     4 |    SKIPPED |    18.351 |       24.732 |         1.35x
Ultra Large  |     2 |   32768 |     4 |        256 |    16 |    SKIPPED |     9.142 |       15.229 |         1.67x
Ultra Large  |     1 |   61440 |     4 |        48* |    64 |    SKIPPED |    64.838 |      443.532 |         6.84x
Ultra Large  |     1 |   61440 |     4 |        60* |    64 |    SKIPPED |    68.296 |      412.292 |         6.04x
Ultra Large  |     1 |   61440 |     4 |        96* |    64 |    SKIPPED |    70.062 |      377.302 |         5.39x
Ultra Large  |     1 |   61440 |     4 |       120* |    64 |    SKIPPED |    71.827 |      296.017 |         4.12x
Ultra Large  |     1 |   61440 |     4 |       192* |    64 |    SKIPPED |    68.677 |      280.063 |         4.08x
Ultra Large  |     1 |   61440 |     4 |       384* |    32 |    SKIPPED |    70.363 |      144.610 |         2.06x
Ultra Large  |     1 |   65536 |     4 |         64 |    64 |    SKIPPED |    74.312 |      464.139 |         6.25x
Ultra Large  |     1 |   65536 |     4 |        128 |    64 |    SKIPPED |    79.931 |      400.425 |         5.01x
Ultra Large  |     1 |   65536 |     4 |        256 |    32 |    SKIPPED |    79.668 |      174.698 |         2.19x
Ultra Large  |     1 |   65536 |     4 |        512 |    32 |    SKIPPED |    75.018 |      169.461 |         2.26x
Ultra Large  |     1 |   65536 |     4 |       1024 |    16 |    SKIPPED |    74.321 |      112.783 |         1.52x
Ultra Large  |     1 |   65536 |     4 |       2048 |     8 |    SKIPPED |    73.620 |      102.813 |         1.40x
Ultra Large  |     1 |   65536 |     4 |       4096 |     8 |    SKIPPED |    73.618 |      102.002 |         1.39x
Ultra Large  |     1 |   65536 |     4 |       8192 |     4 |    SKIPPED |    74.313 |       96.565 |         1.30x
Ultra Large  |     1 |  131072 |     4 |        128 |    64 |    SKIPPED |   294.695 |     1579.531 |         5.36x
Ultra Large  |     1 |  131072 |     4 |        256 |    64 |    SKIPPED |   311.277 |     1321.690 |         4.25x
Ultra Large  |     1 |  131072 |     4 |        512 |    32 |    SKIPPED |   302.553 |      649.482 |         2.15x
Ultra Large  |     1 |  131072 |     4 |       1024 |    32 |    SKIPPED |   289.628 |      641.068 |         2.21x
Ultra Large  |     1 |  131072 |     4 |       2048 |    16 |    SKIPPED |   287.481 |      447.691 |         1.56x
Ultra Large  |     1 |  131072 |     4 |       4096 |     8 |    SKIPPED |   293.320 |      409.810 |         1.40x
Ultra Large  |     1 |  131072 |     4 |       8192 |     8 |    SKIPPED |   292.453 |      410.092 |         1.40x
Ultra Large  |     1 |  262144 |     4 |        256 |    64 |    SKIPPED |  1230.578 |     5763.011 |         4.68x
Ultra Large  |     1 |  262144 |     4 |        512 |    64 |    SKIPPED |  1291.993 |     4934.654 |         3.82x
Ultra Large  |     1 |  262144 |     4 |       1024 |    32 |    SKIPPED |  1275.991 |     2548.500 |         2.00x
Ultra Large  |     1 |  262144 |     4 |       2048 |    32 |    SKIPPED |  1244.544 |     2532.458 |         2.03x
Ultra Large  |     1 |  262144 |     4 |       4096 |    16 |    SKIPPED |  1235.960 |     1944.330 |         1.57x
Ultra Large  |     1 |  262144 |     4 |       8192 |     8 |    SKIPPED |  1239.453 |     1820.249 |         1.47x
Ultra Large  |     1 |  262144 |     4 |      16384 |     4 |    SKIPPED |  1246.636 |     1600.605 |         1.28x
Ultra Large  |     1 |  470016 |     4 |         64 |    32 |    SKIPPED |  4162.493 |     9513.099 |         2.29x
Ultra Large  |     1 |  470016 |     4 |        128 |    32 |    SKIPPED |  4160.899 |     8698.564 |         2.09x
Ultra Large  |     1 |  470016 |     4 |        256 |    32 |    SKIPPED |  4173.293 |     8636.411 |         2.07x
Ultra Large  |     1 |  470016 |     4 |        512 |    32 |    SKIPPED |  4165.206 |     8391.306 |         2.01x
Ultra Large  |     1 |  470016 |     4 |       1024 |    16 |    SKIPPED |  4185.566 |     6856.521 |         1.64x
```

`*` = non-power-of-2 block size · Total: 113 configurations · Errors: 0
