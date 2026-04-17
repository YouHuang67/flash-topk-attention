"""Sparse attention backward pass."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from einops import rearrange


def preprocess_delta(
    o: torch.Tensor,
    do: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """K1: Compute delta = rowwise dot(O, dO).

    Args:
        o: Forward output ``[B, N, C]``.
        do: Upstream gradient ``[B, N, C]``.
        num_heads: Number of attention heads.

    Returns:
        delta: ``[B, H, N]`` float32.
    """
    B, N, C = o.shape
    H = num_heads
    D = C // H
    o_bhnd = o.float().view(B, N, H, D).permute(0, 2, 1, 3)
    do_bhnd = do.float().view(B, N, H, D).permute(0, 2, 1, 3)
    delta = (o_bhnd * do_bhnd).sum(dim=-1)
    return delta


def build_reverse_indices(
    merged_indices: torch.Tensor,
    counts: torch.Tensor,
    num_kv_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K2: Build reverse index from q-block->kv-block to kv-block->q-block.

    Args:
        merged_indices: ``[B, H, QM, S_MAX]`` int32, -1 padding.
        counts: ``[B, H, QM]`` int32.
        num_kv_blocks: Total number of KV blocks (M).

    Returns:
        (reverse_indices, reverse_counts):
            - reverse_indices: ``[B, H, M, QM]`` int32, -1 padding.
            - reverse_counts: ``[B, H, M]`` int32.
    """
    B, H, QM, S_MAX = merged_indices.shape
    M = num_kv_blocks
    device = merged_indices.device

    reverse_indices = torch.full(
        (B, H, M, QM), -1, dtype=torch.int32, device=device,
    )
    reverse_counts = torch.zeros(
        B, H, M, dtype=torch.int32, device=device,
    )

    for b in range(B):
        for h in range(H):
            for qm in range(QM):
                cnt = counts[b, h, qm].item()
                for s in range(cnt):
                    kv_block = merged_indices[b, h, qm, s].item()
                    if kv_block < 0:
                        continue
                    slot = reverse_counts[b, h, kv_block].item()
                    reverse_indices[b, h, kv_block, slot] = qm
                    reverse_counts[b, h, kv_block] += 1

    return reverse_indices, reverse_counts


def sort_by_count(
    reverse_counts: torch.Tensor,
    num_heads: int,
    num_kv_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K4: Sort KV blocks globally by reverse_count descending.

    Args:
        reverse_counts: ``[B, H, M]`` int32.
        num_heads: Number of attention heads.
        num_kv_blocks: Number of KV blocks (M).

    Returns:
        (sorted_counts, sorted_global_ids):
            - sorted_counts: ``[B*H*M]`` int32, descending.
            - sorted_global_ids: ``[B*H*M]`` int32.
    """
    B = reverse_counts.shape[0]
    H = num_heads
    M = num_kv_blocks
    device = reverse_counts.device

    counts_flat = reverse_counts.reshape(-1)
    global_ids = torch.arange(B * H * M, dtype=torch.int32, device=device)

    sorted_order = counts_flat.argsort(descending=True)
    sorted_counts = counts_flat[sorted_order].contiguous()
    sorted_global_ids = global_ids[sorted_order].contiguous()

    return sorted_counts, sorted_global_ids


def bwd_dq_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    merged_indices: torch.Tensor,
    num_heads: int,
    q_block_size: int,
    kv_block_size: int,
    scale: Optional[float] = None,
    q_padding: Tuple[int, int] = (0, 0),
    kv_padding: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Naive dQ computation for testing.

    Args:
        q: ``[B, N, C]`` float32 or bf16.
        k: ``[B, N, C]``.
        v: ``[B, N, C]``.
        do: ``[B, N, C]`` upstream gradient.
        merged_indices: ``[B, H, QM, S_MAX]`` int32.
        num_heads: Number of attention heads.
        q_block_size: Query block size.
        kv_block_size: KV block size.
        scale: Softmax scale.
        q_padding: (q_pad_head, q_pad_tail).
        kv_padding: (kv_pad_head, kv_pad_tail).

    Returns:
        dq: ``[B, N, C]`` float32.
    """
    B, N, C = q.shape
    H = num_heads
    D = C // H

    q_pad_head, _ = q_padding
    kv_pad_head, _ = kv_padding
    N_Q_PADDED = q_padding[0] + N + q_padding[1]
    QM = N_Q_PADDED // q_block_size

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_bhnd = rearrange(q.float(), "b n (h d) -> b h n d", h=H)
    k_bhnd = rearrange(k.float(), "b n (h d) -> b h n d", h=H)
    v_bhnd = rearrange(v.float(), "b n (h d) -> b h n d", h=H)
    do_bhnd = rearrange(do.float(), "b n (h d) -> b h n d", h=H)

    device = q.device
    dq_bhnd = torch.zeros_like(q_bhnd)

    for b in range(B):
        for h in range(H):
            for qm in range(QM):
                q_global_start = qm * q_block_size - q_pad_head

                indices = merged_indices[b, h, qm]
                blocks = indices[indices >= 0]
                if blocks.numel() == 0:
                    continue

                valid = torch.zeros(N, dtype=torch.bool, device=device)
                for bid in blocks.tolist():
                    kv_start = bid * kv_block_size - kv_pad_head
                    kv_end = kv_start + kv_block_size
                    real_start = max(0, kv_start)
                    real_end = min(N, kv_end)
                    if real_start < real_end:
                        valid[real_start:real_end] = True

                if not valid.any():
                    continue

                for row in range(q_block_size):
                    n = q_global_start + row
                    if n < 0 or n >= N:
                        continue

                    s = (q_bhnd[b, h, n] * scale) @ k_bhnd[b, h].T
                    s = s.masked_fill(~valid, float("-inf"))
                    p = torch.softmax(s, dim=-1)

                    dp = do_bhnd[b, h, n] @ v_bhnd[b, h].T
                    delta_i = (do_bhnd[b, h, n] * (p @ v_bhnd[b, h])).sum()
                    ds = p * (dp - delta_i)
                    dq_bhnd[b, h, n] += scale * ds @ k_bhnd[b, h]

    dq = rearrange(dq_bhnd, "b h n d -> b n (h d)")
    return dq


def bwd_dkv_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    merged_indices: torch.Tensor,
    num_heads: int,
    q_block_size: int,
    kv_block_size: int,
    scale: Optional[float] = None,
    q_padding: Tuple[int, int] = (0, 0),
    kv_padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive dK, dV computation for testing.

    Args:
        q: ``[B, N, C]``.
        k: ``[B, N, C]``.
        v: ``[B, N, C]``.
        do: ``[B, N, C]`` upstream gradient.
        merged_indices: ``[B, H, QM, S_MAX]`` int32.
        num_heads: Number of attention heads.
        q_block_size: Query block size.
        kv_block_size: KV block size.
        scale: Softmax scale.
        q_padding: (q_pad_head, q_pad_tail).
        kv_padding: (kv_pad_head, kv_pad_tail).

    Returns:
        (dk, dv): each ``[B, N, C]`` float32.
    """
    B, N, C = q.shape
    H = num_heads
    D = C // H

    q_pad_head, _ = q_padding
    kv_pad_head, _ = kv_padding
    N_Q_PADDED = q_padding[0] + N + q_padding[1]
    QM = N_Q_PADDED // q_block_size

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_bhnd = rearrange(q.float(), "b n (h d) -> b h n d", h=H)
    k_bhnd = rearrange(k.float(), "b n (h d) -> b h n d", h=H)
    v_bhnd = rearrange(v.float(), "b n (h d) -> b h n d", h=H)
    do_bhnd = rearrange(do.float(), "b n (h d) -> b h n d", h=H)

    device = q.device
    dk_bhnd = torch.zeros_like(k_bhnd)
    dv_bhnd = torch.zeros_like(v_bhnd)

    for b in range(B):
        for h in range(H):
            for qm in range(QM):
                q_global_start = qm * q_block_size - q_pad_head

                indices = merged_indices[b, h, qm]
                blocks = indices[indices >= 0]
                if blocks.numel() == 0:
                    continue

                valid = torch.zeros(N, dtype=torch.bool, device=device)
                for bid in blocks.tolist():
                    kv_start = bid * kv_block_size - kv_pad_head
                    kv_end = kv_start + kv_block_size
                    real_start = max(0, kv_start)
                    real_end = min(N, kv_end)
                    if real_start < real_end:
                        valid[real_start:real_end] = True

                if not valid.any():
                    continue

                for row in range(q_block_size):
                    n = q_global_start + row
                    if n < 0 or n >= N:
                        continue

                    s = (q_bhnd[b, h, n] * scale) @ k_bhnd[b, h].T
                    s = s.masked_fill(~valid, float("-inf"))
                    p = torch.softmax(s, dim=-1)

                    dv_bhnd[b, h] += p.unsqueeze(-1) * do_bhnd[b, h, n].unsqueeze(0)

                    dp = do_bhnd[b, h, n] @ v_bhnd[b, h].T
                    delta_i = (do_bhnd[b, h, n] * (p @ v_bhnd[b, h])).sum()
                    ds = p * (dp - delta_i)
                    dk_bhnd[b, h] += scale * ds.unsqueeze(-1) * q_bhnd[b, h, n].unsqueeze(0)

    dk = rearrange(dk_bhnd, "b h n d -> b n (h d)")
    dv = rearrange(dv_bhnd, "b h n d -> b n (h d)")
    return dk, dv


__all__ = [
    "preprocess_delta",
    "build_reverse_indices",
    "sort_by_count",
    "bwd_dq_naive",
    "bwd_dkv_naive",
]
