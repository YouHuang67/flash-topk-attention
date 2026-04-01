[English](README.md) | 中文

# Flash TopK Attention

融合 Flash Attention、top-k KV block 打分与稀疏注意力的 Triton 内核。在计算注意力输出的同时对每个 query 的 KV block 按聚合注意力概率打分，返回 top-k block 索引，用于下游稀疏注意力计算。

## 算法

### 打分（Scoring）

标准 Flash Attention 利用 online softmax 技巧计算 $O = \text{softmax}(QK^\top / \sqrt{D})\ V$，通过逐 block 迭代 KV 避免展开完整的 $N \times N$ 矩阵。其中 $B$ 为 batch size, $N$ 为序列长度, $H$ 为注意力头数, $D$ 为每头维度。

本内核在此基础上将 KV 序列划分为 $M = N / b$ 个不重叠的 block，每个 block 大小为 $b$，并为每个 block $j$ 计算**块级注意力分数**：

$$s_j = \sum_{i \ \in\  \text{block}_j} p_i, \qquad p_i = \frac{\exp(q_t k_i^\top / \sqrt{D})}{\sum_{l=1}^{N} \exp(q_t k_l^\top / \sqrt{D})}.$$

随后选取分数最高的 $\text{top-}k$ 个 block 索引：

$$\mathbf{I} = \underset{j \in \{1,\ldots,M\}}{\text{argtop-}k}\ s_j,$$

在 KV 迭代过程中通过在线 Bitonic Sort 维护 $\text{top-}k$ 池完成选取，无需对 KV 进行第二次遍历。输出 $\mathbf{I}$ 的形状为 $[B, H, N, k]$，可驱动后续稀疏注意力计算，将每 token 的 KV 访问量从 $O(N)$ 降至 $O(k \cdot b)$。

### 稀疏注意力（Sparse Attention）

基于打分阶段得到的 $\text{top-}k$ block 索引 $\mathbf{I}$，`flash_topk_attn` 仅对选中的 KV block 计算注意力。

**Q-Block 共享候选机制**：将 query 按大小 $g$（`q_block_size`）分组。对于第 $m$ 个 query 块（覆盖 query $[mg,\ (m+1)g)$），取组内所有 query 的 top-k 索引的并集，构造共享候选集：

$$\mathcal{C}_m = \bigcup_{t=mg}^{(m+1)g-1} \mathrm{TopK}(q_t)$$

候选按 block id 升序排列，记 $L_m = |\mathcal{C}_m| \leq g \cdot k$。

组内每个 query $q_t$ 对**整个**共享候选集 $\mathcal{C}_m$ 计算注意力：

$$O_t = \sum_{j \in \mathcal{C}_m} \frac{\exp(q_t k_j^\top / \sqrt{D})}{\sum_{j' \in \mathcal{C}_m} \exp(q_t k_{j'}^\top / \sqrt{D})} \cdot v_j$$

- $g=1$：每个 query 仅访问自己的 top-k blocks（per-query 稀疏注意力）
- $g>1$：query 共享候选，实现批量 KV 访问，但每个 query 的注意力范围略有扩大

### 虚拟 Padding

当 $N$ 不能整除 block 大小 $b$ 时，虚拟 padding 将序列扩展为 $N' = \text{pad\_head} + N + \text{pad\_tail}$，满足 $N' \equiv 0 \pmod{b}$。Padding 纯为逻辑概念，QKV 数据不变，padding 位置在 softmax 前被掩码为 $-\infty$。对于头尾的不完整 block，block 分数按有效 token 数归一化：

$$s_j \leftarrow \frac{s_j}{|\{i \in \text{block}_j : i < N\}|}$$

保证完整 block 与部分 block 之间 top-k 排序公平。`flash_topk_attn` 中 Q 侧与 KV 侧 padding 独立指定。

## 性能

基线说明：**Naive** = 标准注意力 + `torch.topk`（两次独立 pass）；**FA2** = 仅 Flash Attention 2 前向，不含 TopK 计分（性能下界）。

| 序列长度 | 注意力头数 | Block 大小 | Top-K | 对比 Naive 加速比 | 对比 FA2 额外开销 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 2,048 | 8 | 64 | 16 | **4.5x 加速** | 1.80x 减速 |
| 4,096 | 8 | 64 | 16 | **5.2x 加速** | 1.96x 减速 |
| 8,192 | 4 | 128 | 16 | **5.3x 加速** | 1.52x 减速 |
| 16,384 | 4 | 256 | 16 | — | 1.58x 减速 |
| 65,536 | 4 | 1,024 | 16 | — | 1.52x 减速 |
| 262,144 | 4 | 4,096 | 16 | — | 1.57x 减速 |

Naive 需要展开完整的 $N \times N$ 注意力矩阵，在长序列下会导致 GPU 显存不足 (OOM)，无法计算 block 分数也无法找到 $\text{top-}k$ 索引。

完整 benchmark 数据：[BENCHMARK_CN.md](BENCHMARK_CN.md)

## 环境要求

- Python >= 3.10
- PyTorch >= 2.5.0, < 2.7.0
- Triton >= 3.0.0
- 支持 CUDA 的 GPU

## 安装

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

## 使用方法

### 打分

```python
import torch
from flash_topk_attn import flash_topk_score

B, N, H, D = 2, 1024, 8, 64

# 3D 输入: [B, N, C]，C = H * D，必须指定 num_heads
q = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)
k = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)
v = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)

o, topk_indices, topk_scores = flash_topk_score(
    q, k, v,
    num_heads=H,
    score_block_size=64,
    topk=16,
)
# o:             [B, N, C]        注意力输出
# topk_indices:  [B, H, N, topk]  int32 block 索引
# topk_scores:   [B, H, N, topk]  float32 block 分数

# 4D 输入: [B, H, N, D]，自动推断 num_heads
q4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
v4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

o4, topk_indices, topk_scores = flash_topk_score(
    q4, k4, v4,
    score_block_size=64,
    topk=16,
)
# o4: [B, H, N, D]  输出与输入 layout 一致
```

**虚拟 padding**（当 `N` 不能整除 `score_block_size` 时）：

```python
# N=1000 不能整除 64，pad 到 1024
o, topk_indices, topk_scores = flash_topk_score(
    q, k, v,
    num_heads=H,
    score_block_size=64,
    topk=16,
    padding=(0, 24),  # pad_head + N + pad_tail 须整除 score_block_size
)
# QKV 数据不变；padding 纯逻辑概念
```

### 稀疏注意力

```python
from flash_topk_attn import flash_topk_score, flash_topk_attn, build_qblock_topk_indices

# 第 1 步：打分，获取每个 query 的 top-k block 索引
o_full, topk_indices, topk_scores = flash_topk_score(
    q, k, v, num_heads=H, score_block_size=64, topk=16,
)
# topk_indices: [B, H, N, topk]

# 第 2 步：构建，按 q-block 聚合 per-query 索引为共享候选集
merged_indices, cu_seqlens = build_qblock_topk_indices(
    topk_indices,       # [B, H, N, topk]
    q_block_size=32,
)
# merged_indices: [B, H, S]      排序去重的 block id，-1 填充
# cu_seqlens:     [B, H, QM+1]   每个 q-block 的累积长度

# 第 3 步：注意力，仅对候选 block 计算稀疏注意力
o_sparse, lse = flash_topk_attn(
    q, k, v,
    merged_indices, cu_seqlens,
    num_heads=H,
    q_block_size=32,
    kv_block_size=64,   # 须与 score_block_size 一致
)
# o_sparse: [B, N, C]    稀疏注意力输出
# lse:      [B, H, N]    log-sum-exp (float32)
```

`merged_indices` 可跨多次注意力调用复用（如多层共享相同稀疏模式）。

**虚拟 padding**（Q 和 KV 独立 padding）：

```python
q_padding  = (8, 24)   # q_pad_head + N + q_pad_tail 须整除 q_block_size
kv_padding = (0, 24)   # kv_pad_head + N + kv_pad_tail 须整除 kv_block_size

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

- [x] Flash Scoring 内核
- [x] 稀疏注意力内核

## 许可证

[MIT](LICENSE)

## 引用

```bibtex
@misc{flash-topk-attention,
  author = {YouHuang67},
  title  = {Flash TopK Attention},
  year   = {2025},
  url    = {https://github.com/YouHuang67/flash-topk-attention},
}
```
