English | [中文](README_CN.md)

# Flash TopK Attention

A fused Triton kernel for Flash Attention with top-k KV block scoring and sparse attention. Computes attention output while scoring each KV block by its aggregated attention probability, returning top-k block indices for downstream sparse attention.

## Algorithm

### Scoring

Standard Flash Attention computes $O = \text{softmax}(QK^\top / \sqrt{D})\ V$ using the online softmax trick, iterating over KV blocks without materializing the full $N \times N$ matrix. Here $B$ is batch size, $N$ is sequence length, $H$ is number of heads, and $D$ is head dimension.

This kernel additionally partitions the KV sequence into $M = N / b$ non-overlapping blocks of size $b$, and computes a **block-level attention score** for each block $j$:

$$s_j = \sum_{i \ \in\  \text{block}_j} p_i, \qquad p_i = \frac{\exp(q_t k_i^\top / \sqrt{D})}{\sum_{l=1}^{N} \exp(q_t k_l^\top / \sqrt{D})}.$$

The $\text{top-}k$ block indices are then selected:

$$\mathbf{I} = \underset{j \in \{1,\ldots,M\}}{\text{argtop-}k}\ s_j,$$

computed by an online Bitonic Sort during the KV iteration with no second pass. The output $\mathbf{I}$ has shape $[B, H, N, k]$ and can drive a subsequent sparse attention pass with $O(k \cdot b)$ KV cost instead of $O(N)$.

### Sparse Attention

Given the top-k block indices $\mathbf{I}$ from scoring, `flash_topk_attn` computes sparse attention over selected KV blocks only.

**Q-Block Shared Candidates**: Queries are grouped into blocks of size $g$ (`q_block_size`). For the $m$-th query block covering queries $[mg,\ (m+1)g)$, we construct a shared candidate set by taking the union of all queries' top-k indices:

$$\mathcal{C}_m = \bigcup_{t=mg}^{(m+1)g-1} \mathrm{TopK}(q_t)$$

The candidates are sorted in ascending order. Let $L_m = |\mathcal{C}_m| \leq g \cdot k$.

Each query $q_t$ in the block attends over the **entire** shared set $\mathcal{C}_m$:

$$O_t = \sum_{j \in \mathcal{C}_m} \frac{\exp(q_t k_j^\top / \sqrt{D})}{\sum_{j' \in \mathcal{C}_m} \exp(q_t k_{j'}^\top / \sqrt{D})} \cdot v_j$$

- When $g=1$: each query attends only to its own top-k blocks (per-query sparse attention)
- When $g>1$: queries share candidates, enabling batched KV access but slightly expanding each query's attention scope

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

### Scoring

```python
import torch
from flash_topk_attn import flash_topk_score

B, N, H, D = 2, 1024, 8, 64

# 3D input: [B, N, C] where C = H * D — num_heads required
q = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)
k = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)
v = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)

o, topk_indices, topk_scores = flash_topk_score(
    q, k, v,
    num_heads=H,
    score_block_size=64,
    topk=16,
)
# o:             [B, N, C]        attention output
# topk_indices:  [B, H, N, topk]  int32 block ids
# topk_scores:   [B, H, N, topk]  float32 block scores

# 4D input: [B, H, N, D] — num_heads inferred from shape
q4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
v4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

o4, topk_indices, topk_scores = flash_topk_score(
    q4, k4, v4,
    score_block_size=64,
    topk=16,
)
# o4: [B, H, N, D] — output matches input layout
```

**Virtual padding** — when `N` is not divisible by `score_block_size`:

```python
# N=1000 is not divisible by 64, pad to 1024
o, topk_indices, topk_scores = flash_topk_score(
    q, k, v,
    num_heads=H,
    score_block_size=64,
    topk=16,
    padding=(0, 24),  # pad_head + N + pad_tail must divide score_block_size
)
# QKV data unchanged; padding is purely logical
```

### Sparse Attention

```python
from flash_topk_attn import flash_topk_score, flash_topk_attn, build_qblock_topk_indices

# Step 1: Score — get per-query top-k block indices
o_full, topk_indices, topk_scores = flash_topk_score(
    q, k, v, num_heads=H, score_block_size=64, topk=16,
)
# topk_indices: [B, H, N, topk]

# Step 2: Build — group per-query indices into per-q-block candidate sets
merged_indices, cu_seqlens = build_qblock_topk_indices(
    topk_indices,       # [B, H, N, topk]
    q_block_size=32,
)
# merged_indices: [B, H, S]      sorted unique block ids, -1 padded
# cu_seqlens:     [B, H, QM+1]   cumulative lengths per q-block

# Step 3: Attend — sparse attention over candidate blocks
o_sparse, lse = flash_topk_attn(
    q, k, v,
    merged_indices, cu_seqlens,
    num_heads=H,
    q_block_size=32,
    kv_block_size=64,   # must match score_block_size
)
# o_sparse: [B, N, C]    sparse attention output
# lse:      [B, H, N]    log-sum-exp (float32)
```

Steps 2 and 3 are decoupled — `merged_indices` can be reused across multiple attention calls (e.g. different layers sharing the same sparsity pattern).

**Virtual padding** for sparse attention (independent Q and KV padding):

```python
q_padding  = (8, 24)   # q_pad_head + N + q_pad_tail must divide q_block_size
kv_padding = (0, 24)   # kv_pad_head + N + kv_pad_tail must divide kv_block_size

merged_indices, cu_seqlens = build_qblock_topk_indices(
    topk_indices, q_block_size=32, q_padding=q_padding,
)
o_sparse, lse = flash_topk_attn(
    q, k, v, merged_indices, cu_seqlens,
    num_heads=H, q_block_size=32, kv_block_size=64,
    q_padding=q_padding, kv_padding=kv_padding,
)
```

## Todo

- [x] Flash Scoring kernel
- [x] Sparse Attention kernel

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
