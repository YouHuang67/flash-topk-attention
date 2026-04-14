"""Top-k block selection from per-query block scores."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _kernel_c_configs(m_pad: int):
    return [
        triton.Config({"Q_BS": q}, num_warps=w)
        for q in [8, 16, 32]
        if q * m_pad * 8 <= 32 * 1024
        for w in [4, 8]
    ]


@lru_cache(maxsize=None)
def _get_topk_select_kernel(m_pad: int):
    configs = _kernel_c_configs(m_pad)

    @triton.autotune(configs=configs, key=["N", "M"])
    @triton.jit
    def _topk_select_kernel(
        BlockScores,
        TopKIndices,
        TopKRawScores,
        TopKAvgScores,
        N: tl.constexpr,
        H: tl.constexpr,
        M: tl.constexpr,
        M_PAD: tl.constexpr,
        M_OUT: tl.constexpr,
        SCORE_BS: tl.constexpr,
        PAD_HEAD: tl.constexpr,
        PAD_TAIL: tl.constexpr,
        Q_BS: tl.constexpr,
        threshold,
        stride_bs_b,
        stride_bs_h,
        stride_bs_n,
        stride_bs_m,
    ):
        i_t = tl.program_id(0)
        i_h = tl.program_id(1)
        i_b = tl.program_id(2)

        q_start = i_t * Q_BS
        q_offs = q_start + tl.arange(0, Q_BS)
        m_offs = tl.arange(0, M_PAD)

        base_in = BlockScores + i_b * stride_bs_b + i_h * stride_bs_h
        ptr_in = base_in + q_offs[:, None] * stride_bs_n + m_offs[None, :] * stride_bs_m
        mask_in = (q_offs[:, None] < N) & (m_offs[None, :] < M)
        raw_scores = tl.load(ptr_in, mask=mask_in, other=0.0)

        valid_count = tl.full((1, M_PAD), SCORE_BS, dtype=tl.float32)
        if PAD_HEAD > 0:
            valid_count = tl.where(m_offs[None, :] == 0, SCORE_BS - PAD_HEAD, valid_count)
        if PAD_TAIL > 0:
            valid_count = tl.where(
                m_offs[None, :] == (M - 1), SCORE_BS - PAD_TAIL, valid_count
            )
        valid_count = tl.where(m_offs[None, :] < M, valid_count, 1.0)
        avg_scores = raw_scores / valid_count

        score_int = avg_scores.to(tl.int32, bitcast=True)
        indices = tl.broadcast_to(m_offs[None, :].to(tl.int32), (Q_BS, M_PAD))
        packed = (score_int.to(tl.int64) << 32) | (indices.to(tl.int64) & 0xFFFFFFFF)
        packed = tl.where(
            m_offs[None, :] < M,
            packed,
            tl.full((Q_BS, M_PAD), -1, dtype=tl.int64),
        )
        sorted_packed = tl.sort(packed, dim=1, descending=True)

        sorted_avg_int = (sorted_packed >> 32).to(tl.int32)
        sorted_avg = sorted_avg_int.to(tl.float32, bitcast=True)
        sorted_idx = sorted_packed.to(tl.int32)

        valid_sorted = m_offs[None, :] < M
        sorted_avg = tl.where(valid_sorted, sorted_avg, 0.0)
        sorted_idx = tl.where(valid_sorted, sorted_idx, -1)

        vc_sorted = tl.full((Q_BS, M_PAD), SCORE_BS, dtype=tl.float32)
        if PAD_HEAD > 0:
            vc_sorted = tl.where(sorted_idx == 0, SCORE_BS - PAD_HEAD, vc_sorted)
        if PAD_TAIL > 0:
            vc_sorted = tl.where(sorted_idx == (M - 1), SCORE_BS - PAD_TAIL, vc_sorted)
        vc_sorted = tl.where(valid_sorted, vc_sorted, 0.0)
        sorted_raw = sorted_avg * vc_sorted

        cs = tl.cumsum(sorted_raw, axis=1)
        exceed = (cs >= threshold).to(tl.int32)
        has_exceed = tl.sum(exceed, axis=1) > 0
        first_exceed = tl.argmax(exceed, axis=1)
        first_exceed = tl.where(has_exceed, first_exceed, M - 1)
        if M_OUT < M:
            first_exceed = tl.minimum(first_exceed, M_OUT - 1)

        keep = m_offs[None, :] <= first_exceed[:, None]
        keep = keep & (m_offs[None, :] < M_OUT) & (q_offs[:, None] < N)

        out_idx = tl.where(keep, sorted_idx, -1)
        out_raw = tl.where(keep, sorted_raw, 0.0)
        out_avg = tl.where(keep, sorted_avg, 0.0)

        out_off = i_b * (H * N * M_OUT) + i_h * (N * M_OUT)
        ptr_out = out_off + q_offs[:, None] * M_OUT + m_offs[None, :]
        mask_out = (q_offs[:, None] < N) & (m_offs[None, :] < M_OUT)
        tl.store(TopKIndices + ptr_out, out_idx, mask=mask_out)
        tl.store(TopKRawScores + ptr_out, out_raw, mask=mask_out)
        tl.store(TopKAvgScores + ptr_out, out_avg, mask=mask_out)

    return _topk_select_kernel


def _flash_topk_select_naive(
    block_scores: torch.Tensor,
    threshold: float,
    max_topk: int,
    score_block_size: int,
    padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation for top-k block selection."""
    if block_scores.ndim != 4:
        raise ValueError("block_scores must be [B, H, N, M]")
    if block_scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("block_scores must be float16/bfloat16/float32")

    M = block_scores.shape[-1]
    pad_head, pad_tail = padding
    if pad_head < 0 or pad_tail < 0:
        raise ValueError("padding must be non-negative (pad_head, pad_tail)")
    if pad_head >= score_block_size or pad_tail >= score_block_size:
        raise ValueError(
            f"padding ({pad_head}, {pad_tail}) must each be < score_block_size "
            f"({score_block_size})"
        )

    raw = block_scores.float()
    valid_count = torch.full((M,), float(score_block_size), device=raw.device, dtype=raw.dtype)
    if pad_head > 0:
        valid_count[0] = float(score_block_size - pad_head)
    if pad_tail > 0:
        valid_count[-1] = float(score_block_size - pad_tail)

    avg = raw / valid_count.view(1, 1, 1, M)
    sorted_avg, sorted_idx = torch.sort(avg, dim=-1, descending=True)
    sorted_raw = torch.gather(raw, dim=-1, index=sorted_idx)

    cs = torch.cumsum(sorted_raw, dim=-1)
    exceed = cs >= threshold
    has_exceed = exceed.any(dim=-1)
    first_exceed = exceed.to(torch.int64).argmax(dim=-1)
    first_exceed = torch.where(
        has_exceed,
        first_exceed,
        torch.full_like(first_exceed, M - 1),
    )

    m_out = min(max_topk, M) if max_topk > 0 else M
    if m_out < M:
        first_exceed = torch.minimum(first_exceed, torch.full_like(first_exceed, m_out - 1))

    pos = torch.arange(M, device=raw.device).view(1, 1, 1, M)
    keep = (pos <= first_exceed.unsqueeze(-1))[..., :m_out]

    topk_indices = torch.where(keep, sorted_idx[..., :m_out], -1).to(torch.int32)
    topk_raw = torch.where(keep, sorted_raw[..., :m_out], 0.0).to(torch.float32)
    topk_avg = torch.where(keep, sorted_avg[..., :m_out], 0.0).to(torch.float32)
    return topk_indices, topk_raw, topk_avg


def flash_topk_select(
    block_scores: torch.Tensor,
    threshold: float,
    max_topk: int,
    score_block_size: int,
    padding: Tuple[int, int] = (0, 0),
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort block scores by avg, cumsum raw scores, and truncate at threshold.

    Args:
        block_scores: Per-query block scores, shape ``[B, H, N, M]``.
            Accepts float32/float16/bfloat16 (internally cast to float32).
        threshold: Cumulative raw-score threshold. Blocks are kept until
            the running sum of sorted raw scores reaches this value.
        max_topk: Maximum number of blocks to keep per query. 0 means
            no limit (keep up to M).
        score_block_size: Number of tokens per score block, used to
            convert raw scores to avg scores (``avg = raw / valid_count``).
        padding: ``(pad_head, pad_tail)`` — number of invalid tokens at
            the first and last block. Reduces their valid_count accordingly.
        backend: Compute backend selection.
            - ``"auto"`` (default): Triton for M_PAD <= 128, CUDA for larger.
            - ``"triton"``: Force Triton (M_PAD must be <= 512).
            - ``"cuda"``: Force CUDA CUB (M must be <= 4096).

    Returns:
        A tuple of three tensors, each ``[B, H, N, M_OUT]``:
            - **topk_indices** (int32): Original block indices in descending
              avg-score order. ``-1`` marks positions beyond the cutoff.
            - **topk_raw_scores** (float32): Corresponding raw scores.
              ``0.0`` beyond the cutoff.
            - **topk_avg_scores** (float32): Corresponding avg scores.
              ``0.0`` beyond the cutoff.

        where ``M_OUT = min(max_topk, M)`` if ``max_topk > 0``, else ``M``.
    """
    if block_scores.ndim != 4:
        raise ValueError("block_scores must be [B, H, N, M]")
    if block_scores.device.type != "cuda":
        raise ValueError("flash_topk_select requires CUDA tensor")
    if block_scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("block_scores must be float16/bfloat16/float32")

    B, H, N, M = block_scores.shape
    if M <= 0:
        raise ValueError("M must be positive")
    if score_block_size <= 0:
        raise ValueError("score_block_size must be positive")

    pad_head, pad_tail = padding
    if pad_head < 0 or pad_tail < 0:
        raise ValueError("padding must be non-negative (pad_head, pad_tail)")
    if pad_head >= score_block_size or pad_tail >= score_block_size:
        raise ValueError(
            f"padding ({pad_head}, {pad_tail}) must each be < score_block_size "
            f"({score_block_size})"
        )
    if max_topk < 0:
        raise ValueError("max_topk must be >= 0")

    m_pad = _next_power_of_2(M)
    if backend == "auto":
        use_cuda = m_pad > 128
    elif backend == "triton":
        if m_pad > 512:
            raise ValueError(
                f"Triton backend requires M_PAD <= 512, got M={M} (M_PAD={m_pad})"
            )
        use_cuda = False
    elif backend == "cuda":
        if M > 4096:
            raise NotImplementedError(
                f"CUDA backend requires M <= 4096, got M={M}"
            )
        use_cuda = True
    else:
        raise ValueError(
            f"backend must be 'auto', 'triton', or 'cuda', got '{backend}'"
        )

    if use_cuda:
        from flash_topk_attn_v2.topk_select_cuda import flash_topk_select_cuda

        return flash_topk_select_cuda(
            block_scores, threshold, max_topk, score_block_size, padding,
        )

    m_out = min(max_topk, M) if max_topk > 0 else M

    topk_indices = torch.full(
        (B, H, N, m_out), -1, device=block_scores.device, dtype=torch.int32,
    )
    topk_raw = torch.zeros(
        (B, H, N, m_out), device=block_scores.device, dtype=torch.float32,
    )
    topk_avg = torch.zeros(
        (B, H, N, m_out), device=block_scores.device, dtype=torch.float32,
    )

    if block_scores.dtype == torch.float32:
        block_scores_in = block_scores
    else:
        block_scores_in = block_scores.float()

    kernel = _get_topk_select_kernel(m_pad)
    grid = lambda meta: (triton.cdiv(N, meta["Q_BS"]), H, B)
    kernel[grid](
        block_scores_in,
        topk_indices,
        topk_raw,
        topk_avg,
        N=N,
        H=H,
        M=M,
        M_PAD=m_pad,
        M_OUT=m_out,
        SCORE_BS=score_block_size,
        PAD_HEAD=pad_head,
        PAD_TAIL=pad_tail,
        threshold=float(threshold),
        stride_bs_b=block_scores_in.stride(0),
        stride_bs_h=block_scores_in.stride(1),
        stride_bs_n=block_scores_in.stride(2),
        stride_bs_m=block_scores_in.stride(3),
    )
    return topk_indices, topk_raw, topk_avg


__all__ = ["flash_topk_select", "_flash_topk_select_naive"]
