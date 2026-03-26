"""
Reverse-index two-stage sparse top-k block attention (forward only).

Stage1: grid (M, D/BV, B*H), local softmax in log2. Stage2: per-query merge of partials.
Plan: docs/plan/20260325_reverse_index_two_stage_attention.md.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flash_topk_attn.scoring import _next_power_of_2


def build_reverse_index(
    block_indices: torch.Tensor, M: int, topk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort flat indices by block id; return sorted_flat_idx and cu_seqlen-style offsets.

    Args:
        block_indices: [B, H_BI, N, topk] int32/int64
        M: num KV blocks = N // block_size
        topk: k per query

    Returns:
        sorted_flat_idx: [B, H_BI, N * topk] int32
        block_offsets: [B, H_BI, M + 1] int32
    """
    B, H_BI, N, k = block_indices.shape
    assert k == topk
    device = block_indices.device
    bi_flat = block_indices.reshape(B, H_BI, N * topk)
    sorted_block_ids, sorted_flat_idx = torch.sort(bi_flat, dim=-1, stable=True)
    boundaries = torch.arange(M + 1, device=device, dtype=sorted_block_ids.dtype)
    block_offsets = torch.searchsorted(
        sorted_block_ids.contiguous(),
        boundaries.view(1, 1, -1).expand(B, H_BI, -1).contiguous(),
    )
    return sorted_flat_idx.to(torch.int32), block_offsets.to(torch.int32)


@triton.jit
def _reverse_attn_stage1_kernel(
    q_ptr, k_ptr, v_ptr,
    sorted_flat_idx_ptr, block_offsets_ptr,
    mid_o_ptr, mid_lse_ptr,
    stride_q_b, stride_q_t, stride_q_h, stride_q_d,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_sf_b, stride_sf_h,
    stride_bo_b, stride_bo_h,
    stride_mo_b, stride_mo_h, stride_mo_t, stride_mo_k, stride_mo_d,
    stride_ml_b, stride_ml_h, stride_ml_t, stride_ml_k,
    B, H, H_BI, M, N, D, topk, scale,
    Q_BS: tl.constexpr,
    SCORE_BS_ORIG: tl.constexpr,
    SCORE_BS: tl.constexpr,
    BD: tl.constexpr, BV: tl.constexpr,
):
    NEG_INF: tl.constexpr = -1e38
    LOG2E: tl.constexpr = 1.4426950408889634

    i_m = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = (i_bh // H).to(tl.int64)
    i_h = (i_bh % H).to(tl.int64)
    i_h_bi = tl.minimum(i_h, H_BI - 1)

    scale_log2 = scale * LOG2E

    bo_ptr = block_offsets_ptr + i_b * stride_bo_b + i_h_bi * stride_bo_h
    start = tl.load(bo_ptr + i_m)
    end = tl.load(bo_ptr + i_m + 1)
    query_count = end - start

    if query_count == 0:
        return

    kv_offset = i_m * SCORE_BS_ORIG

    p_k = tl.make_block_ptr(
        base=k_ptr + i_b * stride_k_b + i_h * stride_k_h,
        shape=(D, N),
        strides=(stride_k_d, stride_k_t),
        offsets=(0, kv_offset),
        block_shape=(BD, SCORE_BS),
        order=(0, 1),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)

    p_v = tl.make_block_ptr(
        base=v_ptr + i_b * stride_v_b + i_h * stride_v_h,
        shape=(N, D),
        strides=(stride_v_t, stride_v_d),
        offsets=(kv_offset, i_v * BV),
        block_shape=(SCORE_BS, BV),
        order=(1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

    kv_j = tl.arange(0, SCORE_BS)
    kv_idx = kv_offset + kv_j
    kv_valid = (kv_j < SCORE_BS_ORIG) & (kv_idx < N)

    sf_ptr = sorted_flat_idx_ptr + i_b * stride_sf_b + i_h_bi * stride_sf_h
    q_base = q_ptr + i_b * stride_q_b + i_h * stride_q_h
    mo_base = mid_o_ptr + i_b * stride_mo_b + i_h * stride_mo_h
    ml_base = mid_lse_ptr + i_b * stride_ml_b + i_h * stride_ml_h

    for q_off in range(0, query_count, Q_BS):
        actual = tl.minimum(Q_BS, query_count - q_off)
        q_mask = tl.arange(0, Q_BS) < actual

        flat_idx = tl.load(
            sf_ptr + start + q_off + tl.arange(0, Q_BS), mask=q_mask, other=0
        )
        q_pos = flat_idx // topk
        slot = flat_idx % topk

        d_idx = tl.arange(0, BD)
        q_offsets = q_pos[:, None] * stride_q_t + d_idx[None, :] * stride_q_d
        b_q = tl.load(
            q_base + q_offsets,
            mask=q_mask[:, None] & (d_idx[None, :] < D),
            other=0.0,
        )
        b_q = b_q.to(tl.float32) * scale_log2

        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(q_mask[:, None] & kv_valid[None, :], b_s, NEG_INF)

        b_m = tl.max(b_s, axis=1)
        b_p = tl.exp2(b_s - b_m[:, None])
        b_l = tl.sum(b_p, axis=1)
        b_o = tl.dot(b_p, b_v) / b_l[:, None]
        b_lse = b_m + tl.log2(b_l)

        v_idx = i_v * BV + tl.arange(0, BV)
        mo_offsets = (
            q_pos[:, None] * stride_mo_t
            + slot[:, None] * stride_mo_k
            + v_idx[None, :] * stride_mo_d
        )
        tl.store(
            mo_base + mo_offsets,
            b_o,
            mask=q_mask[:, None] & (v_idx[None, :] < D),
        )

        if i_v == 0:
            ml_offsets = q_pos * stride_ml_t + slot * stride_ml_k
            tl.store(ml_base + ml_offsets, b_lse, mask=q_mask)


@triton.jit
def _reverse_attn_stage2_kernel(
    mid_o_ptr, mid_lse_ptr, o_ptr, lse_ptr,
    stride_mo_b, stride_mo_h, stride_mo_t, stride_mo_k, stride_mo_d,
    stride_ml_b, stride_ml_h, stride_ml_t, stride_ml_k,
    stride_o_b, stride_o_t, stride_o_h, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_t,
    B, H, N, D, topk,
    BV: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471805599453

    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = (i_bh // H).to(tl.int64)
    i_h = (i_bh % H).to(tl.int64)

    acc_o = tl.zeros([BV], dtype=tl.float32)
    acc_m = tl.full([], float("-inf"), dtype=tl.float32)
    acc_l = tl.zeros([], dtype=tl.float32)

    mo_base = mid_o_ptr + i_b * stride_mo_b + i_h * stride_mo_h + i_t * stride_mo_t
    ml_base = mid_lse_ptr + i_b * stride_ml_b + i_h * stride_ml_h + i_t * stride_ml_t

    for i_k in range(topk):
        p_mo = tl.make_block_ptr(
            base=mo_base + i_k * stride_mo_k,
            shape=(D,),
            strides=(stride_mo_d,),
            offsets=(i_v * BV,),
            block_shape=(BV,),
            order=(0,),
        )
        o_i = tl.load(p_mo, boundary_check=(0,)).to(tl.float32)

        lse_i = tl.load(ml_base + i_k * stride_ml_k)

        new_m = tl.maximum(acc_m, lse_i)
        old_scale = tl.exp2(acc_m - new_m)
        new_scale = tl.exp2(lse_i - new_m)

        acc_o = acc_o * old_scale + o_i * new_scale
        acc_l = acc_l * old_scale + new_scale
        acc_m = new_m

    final_o = acc_o / acc_l

    p_o = tl.make_block_ptr(
        base=o_ptr + i_b * stride_o_b + i_t * stride_o_t + i_h * stride_o_h,
        shape=(D,),
        strides=(stride_o_d,),
        offsets=(i_v * BV,),
        block_shape=(BV,),
        order=(0,),
    )
    tl.store(p_o, final_o.to(o_ptr.dtype.element_ty), boundary_check=(0,))

    if i_v == 0:
        final_lse = (acc_m + tl.log2(acc_l)) * LN2
        tl.store(
            lse_ptr + i_b * stride_lse_b + i_h * stride_lse_h + i_t * stride_lse_t,
            final_lse,
        )


def flash_topk_attn_reverse_index_two_stage(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    num_heads: int,
    block_size: int = 64,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    q_bs: int = 16,
) -> torch.Tensor:
    """Sparse attention over top-k KV blocks (reverse index + two-stage merge).

    Stage1 loads one score block per KV index. The sequence tile for ``block_ptr`` is
    ``next_power_of_2(block_size)`` with validity mask on the true block length (v0-style).
    Stage1 is configured so the padded score tile is always >= 16, so it can use
    ``tl.dot`` for scores and weighted values (Triton dot tile minimum).

    Args:
        q, k, v: [B, N, C], C = num_heads * D
        block_indices: [B, H_BI, N, topk] int32, H_BI in {1, H}
        num_heads: H
        block_size: tokens per KV/score block; must divide N
        scale: attention scale; default 1/sqrt(D)
        num_kv_heads: reserved; must equal num_heads
        q_bs: Stage1 batch size over reverse-index refs (constexpr in kernel)

    Returns:
        o: [B, N, C]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "inputs must be on CUDA"
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    B, N, C = q.shape
    H = num_heads
    D = C // H
    topk = block_indices.shape[-1]
    assert C == H * D, f"C={C} must equal num_heads * D, got H={H}"
    assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"
    M = N // block_size

    if num_kv_heads is not None and num_kv_heads != num_heads:
        raise NotImplementedError("GQA (num_kv_heads != num_heads) is not supported")

    H_BI = block_indices.shape[1]
    assert H_BI == 1 or H_BI == H, (
        f"block_indices.shape[1] must be 1 or num_heads(H={H}), got {H_BI}"
    )

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_indices = block_indices.contiguous().to(torch.int32)

    BD = _next_power_of_2(D)
    BV = min(BD, 64)
    Q_BS = max(16, q_bs)

    q_h = q.view(B, N, H, D)
    k_h = k.view(B, N, H, D)
    v_h = v.view(B, N, H, D)
    o_h = torch.empty_like(q_h)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

    sorted_flat_idx, block_offsets = build_reverse_index(block_indices, M, topk)

    mid_o = torch.empty(B, H, N, topk, D, device=q.device, dtype=torch.float32)
    mid_lse = torch.empty(B, H, N, topk, device=q.device, dtype=torch.float32)

    grid1 = (M, triton.cdiv(D, BV), B * H)
    _reverse_attn_stage1_kernel[grid1](
        q_h, k_h, v_h,
        sorted_flat_idx, block_offsets,
        mid_o, mid_lse,
        *q_h.stride(),
        *k_h.stride(),
        *v_h.stride(),
        sorted_flat_idx.stride(0), sorted_flat_idx.stride(1),
        block_offsets.stride(0), block_offsets.stride(1),
        *mid_o.stride(),
        *mid_lse.stride(),
        B=B, H=H, H_BI=H_BI, M=M, N=N, D=D, topk=topk, scale=scale,
        Q_BS=Q_BS,
        SCORE_BS_ORIG=block_size,
        SCORE_BS=max(16, _next_power_of_2(block_size)),
        BD=BD, BV=BV,
    )

    grid2 = (N, triton.cdiv(D, BV), B * H)
    _reverse_attn_stage2_kernel[grid2](
        mid_o, mid_lse, o_h, lse,
        *mid_o.stride(),
        *mid_lse.stride(),
        *o_h.stride(),
        lse.stride(0), lse.stride(1), lse.stride(2),
        B=B, H=H, N=N, D=D, topk=topk,
        BV=BV,
    )

    return o_h.view(B, N, C)
