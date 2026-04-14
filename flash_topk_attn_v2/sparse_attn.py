"""Sparse attention: block-sparse forward using merged_indices from Kernel D."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from einops import rearrange


def _flash_sparse_attn_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    merged_indices: torch.Tensor,
    num_heads: int,
    q_block_size: int = 64,
    kv_block_size: int = 64,
    scale: Optional[float] = None,
    q_padding: Tuple[int, int] = (0, 0),
    kv_padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for block-sparse attention (v2 pipeline).

    Args:
        q: Query tensor, shape ``[B, N, C]`` where ``C = num_heads * D``.
        k: Key tensor, shape ``[B, N, C]``.
        v: Value tensor, shape ``[B, N, C]``.
        merged_indices: Per-qblock KV block indices from Kernel D,
            shape ``[B, H, QM, qblock_topk]``, int32, ``-1`` for padding,
            score descending order.
        num_heads: Number of attention heads.
        q_block_size: Query block size.
        kv_block_size: KV block size.
        scale: Softmax scale. Defaults to ``1 / sqrt(D)``.
        q_padding: (q_pad_head, q_pad_tail) virtual Q padding.
        kv_padding: (kv_pad_head, kv_pad_tail) virtual KV padding.

    Returns:
        Tuple of (output, lse):
            - output: ``[B, N, C]``, same dtype as q.
            - lse: ``[B, H, N]``, float32 log-sum-exp.
    """
    orig_dtype = q.dtype
    if orig_dtype == torch.float32:
        q = q.bfloat16()
        k = k.bfloat16()
        v = v.bfloat16()
    B, N, C = q.shape
    H = num_heads
    D = C // H

    q_pad_head, q_pad_tail = q_padding
    kv_pad_head, kv_pad_tail = kv_padding
    N_Q_PADDED = q_pad_head + N + q_pad_tail
    N_KV_PADDED = kv_pad_head + N + kv_pad_tail
    QM = N_Q_PADDED // q_block_size

    assert k.shape == (B, N, C) and v.shape == (B, N, C)
    assert C == H * D
    assert N_Q_PADDED % q_block_size == 0
    assert N_KV_PADDED % kv_block_size == 0
    assert merged_indices.shape[:2] == (B, H)
    assert merged_indices.shape[2] == QM

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_bhnd = rearrange(q, "b n (h d) -> b h n d", h=H).float()
    k_bhnd = rearrange(k, "b n (h d) -> b h n d", h=H).float()
    v_bhnd = rearrange(v, "b n (h d) -> b h n d", h=H).float()

    device = q.device
    output = torch.zeros(B, H, N, D, device=device, dtype=torch.float32)
    lse = torch.full(
        (B, H, N), float("-inf"), device=device, dtype=torch.float32
    )

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
                    row_lse = torch.logsumexp(s, dim=-1)
                    p = torch.exp(s - row_lse)
                    output[b, h, n] = p @ v_bhnd[b, h]
                    lse[b, h, n] = row_lse

    output = rearrange(output, "b h n d -> b n (h d)")
    return output.to(orig_dtype), lse


def flash_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    merged_indices: torch.Tensor,
    num_heads: int,
    q_block_size: int = 64,
    kv_block_size: int = 64,
    scale: Optional[float] = None,
    q_padding: Tuple[int, int] = (0, 0),
    kv_padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse attention forward using CUDA kernel.

    Args:
        q: Query tensor, shape ``[B, N, C]`` where ``C = num_heads * D``.
        k: Key tensor, shape ``[B, N, C]``.
        v: Value tensor, shape ``[B, N, C]``.
        merged_indices: Per-qblock KV block indices from Kernel D,
            shape ``[B, H, QM, qblock_topk]``, int32, ``-1`` for padding.
            QM = (q_pad_head + N + q_pad_tail) // q_block_size.
        num_heads: Number of attention heads.
        q_block_size: Query block size (1 to 64).
        kv_block_size: KV block size (any positive integer).
        scale: Softmax scale. Defaults to ``1 / sqrt(D)``.
        q_padding: (q_pad_head, q_pad_tail) virtual Q padding.
        kv_padding: (kv_pad_head, kv_pad_tail) virtual KV padding.

    Returns:
        Tuple of (output, lse):
            - output: ``[B, N, C]``, same dtype as q.
            - lse: ``[B, H, N]``, float32.
    """
    if q.ndim != 3:
        raise ValueError("q must be [B, N, C]")
    if q.device.type != "cuda":
        raise ValueError("flash_sparse_attn requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("q must be float16, bfloat16, or float32")

    orig_dtype = q.dtype
    if orig_dtype == torch.float32:
        q = q.bfloat16()
        k = k.bfloat16()
        v = v.bfloat16()

    B, N, C = q.shape
    H = num_heads
    D = C // H

    if C != H * D:
        raise ValueError(f"C ({C}) must be divisible by num_heads ({H})")
    if q_block_size <= 0 or q_block_size > 64:
        raise ValueError(
            f"q_block_size must be in (0, 64], got {q_block_size}"
        )
    if kv_block_size <= 0:
        raise ValueError(
            f"kv_block_size must be positive, got {kv_block_size}"
        )
    if merged_indices.ndim != 4:
        raise ValueError("merged_indices must be [B, H, QM, qblock_topk]")
    if merged_indices.dtype != torch.int32:
        raise ValueError("merged_indices must be int32")

    q_pad_head, q_pad_tail = q_padding
    kv_pad_head, kv_pad_tail = kv_padding

    if q_pad_head < 0 or q_pad_tail < 0:
        raise ValueError("q_padding values must be non-negative")
    if kv_pad_head < 0 or kv_pad_tail < 0:
        raise ValueError("kv_padding values must be non-negative")
    if q_pad_head >= q_block_size:
        raise ValueError(
            f"q_pad_head={q_pad_head} must be < q_block_size={q_block_size}"
        )
    if q_pad_tail >= q_block_size:
        raise ValueError(
            f"q_pad_tail={q_pad_tail} must be < q_block_size={q_block_size}"
        )
    if kv_pad_head >= kv_block_size:
        raise ValueError(
            f"kv_pad_head={kv_pad_head} must be < "
            f"kv_block_size={kv_block_size}"
        )
    if kv_pad_tail >= kv_block_size:
        raise ValueError(
            f"kv_pad_tail={kv_pad_tail} must be < "
            f"kv_block_size={kv_block_size}"
        )

    N_Q_PADDED = q_pad_head + N + q_pad_tail
    N_KV_PADDED = kv_pad_head + N + kv_pad_tail

    if N_Q_PADDED % q_block_size != 0:
        raise ValueError(
            f"q_pad_head + N + q_pad_tail = {N_Q_PADDED} must be divisible "
            f"by q_block_size={q_block_size}"
        )
    if N_KV_PADDED % kv_block_size != 0:
        raise ValueError(
            f"kv_pad_head + N + kv_pad_tail = {N_KV_PADDED} must be "
            f"divisible by kv_block_size={kv_block_size}"
        )
    min_n = 64
    pad_n_physical = 0
    if N < min_n:
        pad_n_physical = min_n - N
        q = torch.nn.functional.pad(q, (0, 0, 0, pad_n_physical))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_n_physical))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_n_physical))
        N_phys = min_n
    else:
        N_phys = N

    QM = N_Q_PADDED // q_block_size
    if merged_indices.shape[2] != QM:
        raise ValueError(
            f"merged_indices dim 2 ({merged_indices.shape[2]}) must match "
            f"QM = (q_pad_head + N + q_pad_tail) // q_block_size = {QM}"
        )

    _SUPPORTED_D = (32, 64, 96, 128, 160, 256)
    if D <= 0 or D % 16 != 0:
        raise ValueError(f"D ({D}) must be a positive multiple of 16")
    if D > 256:
        raise ValueError(f"D ({D}) must be <= 256")

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    D_kernel = D
    for sd in _SUPPORTED_D:
        if sd >= D:
            D_kernel = sd
            break
    pad_d = D_kernel - D
    if pad_d > 0:
        q = torch.nn.functional.pad(
            q.view(B, N_phys, H, D), (0, pad_d),
        ).reshape(B, N_phys, H * D_kernel)
        k = torch.nn.functional.pad(
            k.view(B, N_phys, H, D), (0, pad_d),
        ).reshape(B, N_phys, H * D_kernel)
        v = torch.nn.functional.pad(
            v.view(B, N_phys, H, D), (0, pad_d),
        ).reshape(B, N_phys, H * D_kernel)

    qblock_topk = merged_indices.shape[3]
    counts = (merged_indices >= 0).sum(dim=-1).int()

    from flash_topk_attn_v2.sparse_attn_cuda import flash_sparse_attn_cuda
    output, lse = flash_sparse_attn_cuda(
        q, k, v, merged_indices, counts,
        num_heads, q_block_size, kv_block_size, qblock_topk, scale,
        q_pad_head, kv_pad_head,
        N_real=N,
    )

    if pad_d > 0:
        output = output.view(
            B, N, H, D_kernel,
        )[..., :D].reshape(B, N, C)
    if orig_dtype == torch.float32:
        output = output.float()
    return output, lse


__all__ = ["flash_sparse_attn", "_flash_sparse_attn_naive"]
