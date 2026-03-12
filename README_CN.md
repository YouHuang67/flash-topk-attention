[English](README.md) | 中文

# Flash TopK Attention

融合 Flash Attention 与 $\text{top-}k$ KV block 打分的 Triton 内核，在计算注意力输出的同时对每个 query 的 KV block 按聚合注意力概率打分，返回 $\text{top-}k$ block 索引，可用于下游稀疏注意力或注意力模式分析。

## 算法

标准 Flash Attention 利用 online softmax 技巧计算 $O = \text{softmax}(QK^\top / \sqrt{D})\ V$，通过逐 block 迭代 KV 避免展开完整的 $N \times N$ 矩阵。其中 $B$ 为 batch size, $N$ 为序列长度, $H$ 为注意力头数, $D$ 为每头维度。

本内核在此基础上将 KV 序列划分为 $M = N / b$ 个不重叠的 block，每个 block 大小为 $b$，并为每个 block $j$ 计算**块级注意力分数**：

$$s_j = \sum_{i \ \in\  \text{block}_j} p_i, \qquad p_i = \frac{\exp(q_t k_i^\top / \sqrt{D})}{\sum_{l=1}^{N} \exp(q_t k_l^\top / \sqrt{D})}.$$

随后选取分数最高的 $\text{top-}k$ 个 block 索引：

$$\mathbf{I} = \underset{j \in \{1,\ldots,M\}}{\text{argtop-}k}\ s_j,$$

在 KV 迭代过程中通过在线 Bitonic Sort 维护 $\text{top-}k$ 池完成选取，无需对 KV 进行第二次遍历。输出 $\mathbf{I}$ 的形状为 $[B, H, N, k]$，可驱动后续稀疏注意力计算，将每 token 的 KV 访问量从 $O(N)$ 降至 $O(k \cdot b)$。

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

Naive 需要展开完整的 $N \times N$ 注意力矩阵，在长序列下会导致 GPU 显存不足 (OOM)，既无法计算 block 分数也无法找到 $\text{top-}k$ 索引。

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
    score_block_size=64,  # b: 每个 KV block 的 token 数，必须整除 N
    topk=16,              # k: 每个 query 选取的 top block 数，须满足 k <= N // b
)
# o:             [B, N, C]
# topk_indices:  [B, H, N, 16]  int32，按分数降序排列
# topk_scores:   [B, H, N, 16]  float32，归一化的 block 注意力权重
```

## Todo

- [x] Flash Scoring 内核（前向 + 反向）
- [ ] 稀疏注意力内核

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
