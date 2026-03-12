English | [中文](README_CN.md)

# Flash TopK Attention

A fused Flash Attention kernel that computes attention output while scoring each KV block by its aggregated attention probability per query, returning the $\text{top-}k$ block indices for downstream sparse attention or attention pattern analysis.

## Algorithm

Standard Flash Attention computes $O = \text{softmax}(QK^\top / \sqrt{D})\ V$ using the online softmax trick, iterating over KV blocks without materializing the full $N \times N$ matrix. Here $B$ is batch size, $N$ is sequence length, $H$ is number of heads, and $D$ is head dimension.

This kernel additionally partitions the KV sequence into $M = N / b$ non-overlapping blocks of size $b$, and computes a **block-level attention score** for each block $j$:

$$s_j = \sum_{i \ \in\  \text{block}_j} p_i, \qquad p_i = \frac{\exp(q_t k_i^\top / \sqrt{D})}{\sum_{l=1}^{N} \exp(q_t k_l^\top / \sqrt{D})}.$$

The $\text{top-}k$ block indices are then selected:

$$\mathbf{I} = \underset{j \in \{1,\ldots,M\}}{\text{argtop-}k}\ s_j,$$

computed by an online Bitonic Sort during the KV iteration with no second pass. The output $\mathbf{I}$ has shape $[B, H, N, k]$ and can drive a subsequent sparse attention pass with $O(k \cdot b)$ KV cost instead of $O(N)$.

## Performance

Baselines: **Naive** = standard attention + `torch.topk` (two separate passes); **FA2** = Flash Attention 2 forward only, no TopK scoring (lower bound).

| Sequence Length | Heads | Block Size | Top-K | Speedup vs Naive | Overhead vs FA2 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 2,048 | 8 | 64 | 16 | **4.5x faster** | 1.80x slower |
| 4,096 | 8 | 64 | 16 | **5.2x faster** | 1.96x slower |
| 8,192 | 4 | 128 | 16 | **5.3x faster** | 1.52x slower |
| 16,384 | 4 | 256 | 16 | — | 1.58x slower |
| 65,536 | 4 | 1,024 | 16 | — | 1.52x slower |
| 262,144 | 4 | 4,096 | 16 | — | 1.57x slower |

Naive materializes the full $N \times N$ attention matrix, causing GPU out-of-memory (OOM) at long sequences — computing block scores or finding $\text{top-}k$ indices becomes impossible.

Full benchmark results: [BENCHMARK.md](BENCHMARK.md)

## Requirements

- Python >= 3.10
- PyTorch >= 2.5.0, < 2.7.0
- Triton >= 3.0.0
- CUDA-capable GPU

## Install

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

## Usage

```python
import torch
from flash_topk_attn import flash_topk_score

B, N, H, D = 2, 1024, 8, 64
C = H * D
q = torch.randn(B, N, C, device="cuda", dtype=torch.float16)
k = torch.randn(B, N, C, device="cuda", dtype=torch.float16)
v = torch.randn(B, N, C, device="cuda", dtype=torch.float16)

o, topk_indices, topk_scores = flash_topk_score(
    q, k, v,
    num_heads=H,
    score_block_size=64,  # b: tokens per KV block, must divide N evenly
    topk=16,              # k: top blocks per query, must satisfy k <= N // b
)
# o:             [B, N, C]
# topk_indices:  [B, H, N, 16]  int32, descending by score
# topk_scores:   [B, H, N, 16]  float32, normalized block attention weights
# Supported dtypes: float16, bfloat16, float32
```

## Todo

- [x] Flash Scoring kernel (forward + backward)
- [ ] Sparse Attention kernel

## License

[MIT](LICENSE)

## Citation

```bibtex
@misc{flash-topk-attention,
  author = {YouHuang67},
  title  = {Flash TopK Attention},
  year   = {2025},
  url    = {https://github.com/YouHuang67/flash-topk-attention},
}
```
