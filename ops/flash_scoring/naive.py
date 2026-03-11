from typing import Optional, Tuple

import torch
from einops import rearrange


def flash_scoring_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    block_size: int,
    topk: int,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash Attention with TopK Block Scoring (naive implementation).

    Args:
        q: Query tensor [B, N, C] where C = num_heads * D.
        k: Key tensor [B, N, C].
        v: Value tensor [B, N, C].
        num_heads: Number of attention heads.
        block_size: Number of tokens per block for score aggregation.
        topk: Number of top blocks to select.
        scale: Attention scale factor, defaults to 1/sqrt(D).

    Returns:
        o: Attention output [B, H, N, D].
        topk_indices: TopK block indices [B, H, N, K].
        topk_scores: TopK block scores [B, H, N, K].
    """
    dtype = q.dtype
    B, N, C = q.shape
    H = num_heads
    D = C // H
    M = N // block_size

    assert k.shape == (B, N, C), f"k shape {k.shape} != {(B, N, C)}"
    assert v.shape == (B, N, C), f"v shape {v.shape} != {(B, N, C)}"
    assert C % H == 0, f"C={C} not divisible by num_heads={H}"
    assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"
    assert topk <= M, f"topk={topk} must be <= M={M}"

    if scale is None:
        scale = D ** -0.5

    q = rearrange(q, 'b n (h d) -> b h n d', h=H)
    k = rearrange(k, 'b n (h d) -> b h n d', h=H)
    v = rearrange(v, 'b n (h d) -> b h n d', h=H)

    q, k, v = map(lambda x: x.float(), (q, k, v))

    scores = torch.einsum('bhqd,bhkd->bhqk', q * scale, k)
    attn_weights = scores.softmax(dim=-1)
    o = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

    attn_blocks = rearrange(attn_weights, 'b h q (m bs) -> b h q m bs', bs=block_size)
    block_scores = attn_blocks.sum(dim=-1)

    topk_scores, topk_indices = torch.topk(
        block_scores, k=topk, dim=-1, largest=True, sorted=True
    )

    return o.to(dtype), topk_indices, topk_scores
