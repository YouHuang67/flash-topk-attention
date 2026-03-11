# Flash TopK Attention

A fused Flash Attention kernel that computes attention output while scoring each KV block by its aggregated attention probability per query, returning the top-$k$ block indices for downstream sparse attention or attention pattern analysis.

## Algorithm

Standard Flash Attention computes $O = \text{softmax}(QK^\top / \sqrt{D})\,V$ using the online softmax trick, iterating over KV blocks without materializing the full $N \times N$ matrix.

This kernel additionally partitions the KV sequence into $M = N / b$ non-overlapping blocks of size $b$, and computes a **block-level attention score** for each block $j$:

$$s_j = \sum_{i \,\in\, \text{block}_j} p_i, \qquad p_i = \frac{\exp(q_t k_i^\top / \sqrt{D})}{\sum_{l=1}^{N} \exp(q_t k_l^\top / \sqrt{D})}$$

The top-$k$ block indices are then selected:

$$\mathcal{I} = \underset{j \in \{1,\ldots,M\}}{\operatorname{argtop-}k}\, s_j$$

computed by an online Bitonic Sort during the KV iteration with no second pass. The output $\mathcal{I}$ has shape $[B, H, N, k]$ and can drive a subsequent sparse attention pass with $O(k \cdot b)$ KV cost instead of $O(N)$.

See [algorithm.md](algorithm.md) for a detailed derivation combining Flash Attention 2 with the single-pass block scoring mechanism.

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

Naive materializes the full $N \times N$ attention matrix, making it infeasible at long sequences where computing block scores or finding top-$k$ indices is impossible within memory limits.

Full benchmark results: [benchmark.md](benchmark.md)

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
from ops.flash_scoring import flash_scoring_triton

B, N, H, D = 2, 1024, 8, 64
C = H * D
q = torch.randn(B, N, C, device="cuda", dtype=torch.float16)
k = torch.randn(B, N, C, device="cuda", dtype=torch.float16)
v = torch.randn(B, N, C, device="cuda", dtype=torch.float16)

o, topk_indices, topk_scores = flash_scoring_triton(
    q, k, v,
    num_heads=H,
    score_block_size=64,  # b: tokens per KV block, must divide N evenly
    topk=16,              # k: top blocks per query, must satisfy k <= N // b
)
# o:             [B, N, C]
# topk_indices:  [B, H, N, 16]  int32, descending by score
# topk_scores:   [B, H, N, 16]  float32, normalized block attention weights
```

## Todo

- [x] Flash Scoring kernel (forward + backward)
- [ ] Sparse Attention kernel

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{flash-topk-attention,
  author = {YouHuang67},
  title  = {Flash TopK Attention},
  year   = {2025},
  url    = {https://github.com/YouHuang67/flash-topk-attention},
}
```
