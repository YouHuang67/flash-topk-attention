"""
Q-block shared KV-candidate sparse attention: exact union of per-query top-k blocks
within each q_block_size group; all queries in the block attend over the same merged set.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

from flash_topk_attn.scoring import _next_power_of_2


def build_qblock_topk_indices(
    topk_indices: torch.Tensor,
    q_block_size: int,
    q_padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sorted unique block ids per q-block (exact union).

    Args:
        topk_indices: [B, H, N, KQ] int32 (or int64); -1 marks empty slots.
        q_block_size: query block size.
        q_padding: (q_pad_head, q_pad_tail) virtual padding for Q alignment.
            q_pad_head + N + q_pad_tail must be divisible by q_block_size.

    Returns:
        merged_indices: [B, H, N * KQ] int32, valid prefix then -1 padding.
        qblock_cu_seqlens: [B, H, QM + 1] int32 cumulative lengths per q-block.
    """
    B, H, N, KQ = topk_indices.shape
    q_pad_head, q_pad_tail = q_padding

    if q_pad_head > 0 or q_pad_tail > 0:
        # pad along sequence dimension: [B, H, N, KQ] -> [B, H, N_Q_PADDED, KQ]
        topk_indices = F.pad(topk_indices, (0, 0, q_pad_head, q_pad_tail), value=-1)
        N_Q_PADDED = q_pad_head + N + q_pad_tail
    else:
        N_Q_PADDED = N

    assert (
        N_Q_PADDED % q_block_size == 0
    ), f"N_Q_PADDED={N_Q_PADDED} must be divisible by q_block_size={q_block_size}"
    QM = N_Q_PADDED // q_block_size
    device = topk_indices.device

    ids = topk_indices.to(torch.int32).view(B, H, QM, q_block_size * KQ)
    ids_sorted, _ = torch.sort(ids, dim=-1, stable=True)

    valid = ids_sorted >= 0
    prev = torch.roll(ids_sorted, shifts=1, dims=-1)
    prev[..., 0] = -2
    is_new = valid & (ids_sorted != prev)

    qblock_lens = is_new.sum(dim=-1, dtype=torch.int32)
    qblock_cu_seqlens = torch.zeros(B, H, QM + 1, device=device, dtype=torch.int32)
    qblock_cu_seqlens[..., 1:] = torch.cumsum(qblock_lens, dim=-1)

    local_offset = torch.cumsum(is_new.to(torch.int32), dim=-1) - 1
    base = qblock_cu_seqlens[..., :-1, None]
    global_idx = base + local_offset

    max_sum = int(qblock_cu_seqlens[:, :, -1].max().item())
    merged_indices = torch.full((B, H, max_sum), -1, device=device, dtype=torch.int32)

    bh_offset = torch.arange(B * H, device=device, dtype=torch.int64).view(B, H, 1, 1)
    global_idx_with_bh = bh_offset * max_sum + global_idx.to(torch.int64)
    is_new_flat = is_new.reshape(-1)
    ids_sorted_flat = ids_sorted.reshape(-1)
    global_idx_flat = global_idx_with_bh.reshape(-1)

    merged_flat = merged_indices.reshape(-1)
    merged_flat[global_idx_flat[is_new_flat]] = ids_sorted_flat[is_new_flat]

    merged_full = torch.full((B, H, N * KQ), -1, device=device, dtype=torch.int32)
    merged_full[:, :, :max_sum] = merged_indices
    return merged_full, qblock_cu_seqlens


@triton.jit
def _qblock_accum_kv(
    kv_block, kv_sub_iter, k_base, v_base, b_q,
    stride_k_d, stride_k_t, stride_v_t, stride_v_d, i_v, N,
    b_m, b_l, b_o,
    KV_BS: tl.constexpr, KV_TILE: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, D: tl.constexpr,
    BRANCH_ID: tl.constexpr, TAIL_LEN: tl.constexpr, NEG_INF_C: tl.constexpr,
    KV_PAD_HEAD: tl.constexpr,
):
    """One KV sub-step: load K/V, tl.dot scores, online softmax in log2 domain."""
    if BRANCH_ID == 1:
        # KV_TILE >= KV_BS: semantic block (KV_BS) with physical tile KV_TILE (2^k).
        token_offset = kv_block * KV_BS - KV_PAD_HEAD

        if KV_PAD_HEAD > 0:
            # Safe path: keep base, shape=(D, N), allow negative offsets guarded by boundary_check.
            p_k = tl.make_block_ptr(
                base=k_base,
                shape=(D, N),
                strides=(stride_k_d, stride_k_t),
                offsets=(0, token_offset),
                block_shape=(BD, KV_TILE),
                order=(0, 1),
            )
            p_v = tl.make_block_ptr(
                base=v_base,
                shape=(N, D),
                strides=(stride_v_t, stride_v_d),
                offsets=(token_offset, i_v * BV),
                block_shape=(KV_TILE, BV),
                order=(1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

            kv_idx = tl.arange(0, KV_TILE)
            tile_valid = (kv_idx < KV_BS) & (token_offset + kv_idx >= 0) & (token_offset + kv_idx < N)
        else:
            # Original fast path: absorb offset into base, logical shape caps to KV_BS.
            p_k = tl.make_block_ptr(
                base=k_base + token_offset * stride_k_t,
                shape=(D, KV_BS),
                strides=(stride_k_d, stride_k_t),
                offsets=(0, 0),
                block_shape=(BD, KV_TILE),
                order=(0, 1),
            )
            p_v = tl.make_block_ptr(
                base=v_base + token_offset * stride_v_t,
                shape=(KV_BS, D),
                strides=(stride_v_t, stride_v_d),
                offsets=(0, i_v * BV),
                block_shape=(KV_TILE, BV),
                order=(1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

            kv_idx = tl.arange(0, KV_TILE)
            tile_valid = (kv_idx < KV_BS) & (token_offset + kv_idx < N)
    else:
        token_offset = kv_block * KV_BS - KV_PAD_HEAD + kv_sub_iter * KV_TILE
        p_k = tl.make_block_ptr(
            base=k_base,
            shape=(D, N),
            strides=(stride_k_d, stride_k_t),
            offsets=(0, token_offset),
            block_shape=(BD, KV_TILE),
            order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            base=v_base,
            shape=(N, D),
            strides=(stride_v_t, stride_v_d),
            offsets=(token_offset, i_v * BV),
            block_shape=(KV_TILE, BV),
            order=(1, 0),
        )
        if D % BD == 0:
            b_k = tl.load(p_k, boundary_check=(1,)).to(tl.float32)
        else:
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        if D % BV == 0:
            b_v = tl.load(p_v, boundary_check=(0,)).to(tl.float32)
        else:
            b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

        kv_idx = token_offset + tl.arange(0, KV_TILE)
        in_block = kv_idx < kv_block * KV_BS - KV_PAD_HEAD + KV_BS
        if KV_PAD_HEAD > 0:
            base_valid = (kv_idx >= 0) & (kv_idx < N) & in_block
        else:
            base_valid = (kv_idx < N) & in_block
        if BRANCH_ID == 3:
            tile_valid = base_valid & (tl.arange(0, KV_TILE) < TAIL_LEN)
        else:
            tile_valid = base_valid

    b_s = tl.dot(b_q, b_k)
    b_s = tl.where(tile_valid[None, :], b_s, NEG_INF_C)

    b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
    scale_old = tl.exp2(b_m - b_m_new)
    b_p = tl.exp2(b_s - b_m_new[:, None])
    b_l = b_l * scale_old + tl.sum(b_p, axis=1)
    b_o = b_o * scale_old[:, None] + tl.dot(b_p, b_v)
    b_m = b_m_new
    return b_m, b_l, b_o


@triton.autotune(
    configs=[
        triton.Config({"Q_TILE": q, "KV_TILE": kv}, num_warps=w, num_stages=s)
        for q in (16, 32)
        for kv in (16, 32, 64)
        for w in (4, 8)
        for s in (2, 3)
    ],
    key=["Q_BS", "KV_BS", "D"],
)
@triton.jit
def _flash_topk_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, merged_indices_ptr, qblock_cu_seqlens_ptr, o_ptr, lse_ptr,
    stride_q_b, stride_q_t, stride_q_h, stride_q_d,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_mi_b, stride_mi_h, stride_mi_s,
    stride_cu_b, stride_cu_h, stride_cu_q,
    stride_o_b, stride_o_t, stride_o_h, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_t,
    B, H, H_BI, N, D, QM, scale,
    Q_BS: tl.constexpr, KV_BS: tl.constexpr, Q_TILE: tl.constexpr, KV_TILE: tl.constexpr,
    BD: tl.constexpr, BV: tl.constexpr,
    Q_PAD_HEAD: tl.constexpr, KV_PAD_HEAD: tl.constexpr,
):
    NEG_INF_C: tl.constexpr = -1e38

    if KV_TILE >= KV_BS:
        BRANCH_ID: tl.constexpr = 1
        INNER_FULL: tl.constexpr = 0
        INNER_TAIL: tl.constexpr = 0
        TAIL_LEN: tl.constexpr = 0
    else:
        INNER_FULL: tl.constexpr = KV_BS // KV_TILE
        TAIL_LEN: tl.constexpr = KV_BS - INNER_FULL * KV_TILE
        if TAIL_LEN == 0:
            BRANCH_ID: tl.constexpr = 2
            INNER_TAIL: tl.constexpr = 0
        else:
            BRANCH_ID: tl.constexpr = 3
            INNER_TAIL: tl.constexpr = 1

    NUM_Q_TILES: tl.constexpr = (Q_BS + Q_TILE - 1) // Q_TILE
    INNER_ITERS: tl.constexpr = INNER_FULL + INNER_TAIL
    USE_STATIC_INNER: tl.constexpr = INNER_ITERS <= 16

    i_q = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    i_h_bi = tl.minimum(i_h, H_BI - 1)

    qm = i_q // NUM_Q_TILES
    q_tile_i = i_q % NUM_Q_TILES
    q_subtile_offset = qm * Q_BS + q_tile_i * Q_TILE - Q_PAD_HEAD

    LOG2E_C: tl.constexpr = 1.4426950408889634
    scale_log2 = scale * LOG2E_C

    q_base = q_ptr + i_b * stride_q_b + i_h * stride_q_h
    k_base = k_ptr + i_b * stride_k_b + i_h * stride_k_h
    v_base = v_ptr + i_b * stride_v_b + i_h * stride_v_h

    p_q = tl.make_block_ptr(
        base=q_base,
        shape=(N, D),
        strides=(stride_q_t, stride_q_d),
        offsets=(q_subtile_offset, 0),
        block_shape=(Q_TILE, BD),
        order=(1, 0),
    )
    if Q_PAD_HEAD > 0:
        # With Q padding we may read outside [0, N); rely on boundary_check
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * scale_log2
    else:
        if Q_BS % Q_TILE == 0 and D % BD == 0:
            b_q = tl.load(p_q).to(tl.float32) * scale_log2
        elif Q_BS % Q_TILE == 0:
            b_q = tl.load(p_q, boundary_check=(1,)).to(tl.float32) * scale_log2
        elif D % BD == 0:
            b_q = tl.load(p_q, boundary_check=(0,)).to(tl.float32) * scale_log2
        else:
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * scale_log2

    q_row = q_subtile_offset + tl.arange(0, Q_TILE)
    if Q_PAD_HEAD > 0:
        q_valid = (q_row >= 0) & (q_row < N)
    else:
        q_valid = (q_row < N) & (q_row < qm * Q_BS + Q_BS)
    b_q = tl.where(q_valid[:, None], b_q, 0.0)

    b_m = tl.full([Q_TILE], float("-inf"), dtype=tl.float32)
    b_l = tl.zeros([Q_TILE], dtype=tl.float32)
    b_o = tl.zeros([Q_TILE, BV], dtype=tl.float32)

    cu_ptr = qblock_cu_seqlens_ptr + i_b * stride_cu_b + i_h_bi * stride_cu_h
    start = tl.load(cu_ptr + qm * stride_cu_q).to(tl.int32)
    end = tl.load(cu_ptr + (qm + 1) * stride_cu_q).to(tl.int32)

    mi_row = merged_indices_ptr + i_b * stride_mi_b + i_h_bi * stride_mi_h

    for idx in range(start, end):
        kv_block = tl.load(mi_row + idx * stride_mi_s).to(tl.int32)

        if BRANCH_ID == 1:
            b_m, b_l, b_o = _qblock_accum_kv(
                kv_block, 0, k_base, v_base, b_q,
                stride_k_d, stride_k_t, stride_v_t, stride_v_d, i_v, N, b_m, b_l, b_o,
                KV_BS=KV_BS, KV_TILE=KV_TILE, BD=BD, BV=BV, D=D,
                BRANCH_ID=1, TAIL_LEN=0, NEG_INF_C=NEG_INF_C,
                KV_PAD_HEAD=KV_PAD_HEAD,
            )
        else:
            if USE_STATIC_INNER:
                for kv_sub_iter in tl.static_range(INNER_FULL):
                    b_m, b_l, b_o = _qblock_accum_kv(
                        kv_block, kv_sub_iter, k_base, v_base, b_q,
                        stride_k_d, stride_k_t, stride_v_t, stride_v_d, i_v, N, b_m, b_l, b_o,
                        KV_BS=KV_BS, KV_TILE=KV_TILE, BD=BD, BV=BV, D=D,
                        BRANCH_ID=2, TAIL_LEN=0, NEG_INF_C=NEG_INF_C,
                        KV_PAD_HEAD=KV_PAD_HEAD,
                    )
            else:
                for kv_sub_iter in range(INNER_FULL):
                    b_m, b_l, b_o = _qblock_accum_kv(
                        kv_block, kv_sub_iter, k_base, v_base, b_q,
                        stride_k_d, stride_k_t, stride_v_t, stride_v_d, i_v, N, b_m, b_l, b_o,
                        KV_BS=KV_BS, KV_TILE=KV_TILE, BD=BD, BV=BV, D=D,
                        BRANCH_ID=2, TAIL_LEN=0, NEG_INF_C=NEG_INF_C,
                        KV_PAD_HEAD=KV_PAD_HEAD,
                    )
            if INNER_TAIL:
                b_m, b_l, b_o = _qblock_accum_kv(
                    kv_block, INNER_FULL, k_base, v_base, b_q,
                    stride_k_d, stride_k_t, stride_v_t, stride_v_d, i_v, N, b_m, b_l, b_o,
                    KV_BS=KV_BS, KV_TILE=KV_TILE, BD=BD, BV=BV, D=D,
                    BRANCH_ID=3, TAIL_LEN=TAIL_LEN, NEG_INF_C=NEG_INF_C,
                    KV_PAD_HEAD=KV_PAD_HEAD,
                )

    safe_l = tl.where(b_l > 0.0, b_l, 1.0)
    final_o = b_o / safe_l[:, None]

    o_row = q_subtile_offset + tl.arange(0, Q_TILE)
    if Q_PAD_HEAD > 0:
        o_valid = (o_row >= 0) & (o_row < N)
    else:
        o_valid = q_valid
    v_c = i_v * BV + tl.arange(0, BV)
    o_base = o_ptr + i_b * stride_o_b + i_h * stride_o_h
    o_offs = o_row[:, None] * stride_o_t + v_c[None, :] * stride_o_d
    tl.store(
        o_base + o_offs,
        final_o.to(o_ptr.dtype.element_ty),
        mask=o_valid[:, None] & (v_c[None, :] < D),
    )

    if i_v == 0:
        LN2_C: tl.constexpr = 0.6931471805599453
        final_lse = (b_m + tl.log2(safe_l)) * LN2_C
        lse_base = lse_ptr + i_b * stride_lse_b + i_h * stride_lse_h
        tl.store(
            lse_base + q_row * stride_lse_t,
            final_lse.to(lse_ptr.dtype.element_ty),
            mask=o_valid,
        )


def flash_topk_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    merged_indices: torch.Tensor,
    qblock_cu_seqlens: torch.Tensor,
    num_heads: int,
    q_block_size: int,
    kv_block_size: int,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    kv_padding: Tuple[int, int] = (0, 0),
    q_padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention with exact per-q-block union of top-k KV blocks.

    Args:
        q, k, v: [B, N, C], C = num_heads * D.
        merged_indices: [B, H, S] int32, valid prefix then -1 padding; built by
            :func:`build_qblock_topk_indices`.
        qblock_cu_seqlens: [B, H, QM + 1] int32 cumulative lengths per q-block; built
            by :func:`build_qblock_topk_indices`.
        num_heads: H.
        q_block_size: queries per shared-candidate group.
        kv_block_size: tokens per block id; must match scoring.
        scale: defaults to 1/sqrt(D).
        num_kv_heads: v1 requires num_kv_heads == num_heads.
        kv_padding: (kv_pad_head, kv_pad_tail) virtual padding for KV alignment.
        q_padding: (q_pad_head, q_pad_tail) virtual padding for Q alignment; must
            match the value used when building ``merged_indices``.

    Returns:
        o: [B, N, C], lse: [B, H, N] float32 (natural log).
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    B, N, C = q.shape
    H = num_heads
    D = C // H
    Q_BS, KV_BS = q_block_size, kv_block_size
    assert C == H * D

    if num_kv_heads is not None and num_kv_heads != H:
        raise NotImplementedError("GQA (num_kv_heads != num_heads) is not supported in v1")

    kv_pad_head, kv_pad_tail = kv_padding
    q_pad_head, q_pad_tail = q_padding

    N_KV_PADDED = kv_pad_head + N + kv_pad_tail
    N_Q_PADDED = q_pad_head + N + q_pad_tail

    if N_KV_PADDED % KV_BS != 0:
        need = KV_BS - (N_KV_PADDED % KV_BS)
        raise ValueError(
            f"N_KV_PADDED={N_KV_PADDED} must be divisible by kv_block_size={KV_BS}; "
            f"add {need} tokens of KV padding or adjust kv_block_size."
        )
    if N_Q_PADDED % Q_BS != 0:
        need = Q_BS - (N_Q_PADDED % Q_BS)
        raise ValueError(
            f"N_Q_PADDED={N_Q_PADDED} must be divisible by q_block_size={Q_BS}; "
            f"add {need} tokens of Q padding or adjust q_block_size."
        )

    QM = N_Q_PADDED // Q_BS

    # H_BI is implicit in merged_indices/qblock_cu_seqlens shapes; enforce H_BI in {1, H}.
    H_BI = merged_indices.shape[1]
    assert H_BI == 1 or H_BI == H

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_h = q.contiguous().view(B, N, H, D)
    k_h = k.contiguous().view(B, N, H, D)
    v_h = v.contiguous().view(B, N, H, D)
    o_h = torch.empty_like(q_h)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

    BD = min(128, _next_power_of_2(D))
    BV = min(BD, 64)

    grid = lambda meta: (
        QM * triton.cdiv(Q_BS, meta["Q_TILE"]),
        triton.cdiv(D, BV),
        B * H,
    )

    _flash_topk_attn_fwd_kernel[grid](
        q_h,
        k_h,
        v_h,
        merged_indices,
        qblock_cu_seqlens,
        o_h,
        lse,
        *q_h.stride(),
        *k_h.stride(),
        *v_h.stride(),
        *merged_indices.stride(),
        *qblock_cu_seqlens.stride(),
        *o_h.stride(),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        B,
        H,
        H_BI,
        N,
        D,
        QM,
        scale,
        Q_BS=Q_BS,
        KV_BS=KV_BS,
        BD=BD,
        BV=BV,
        Q_PAD_HEAD=q_pad_head,
        KV_PAD_HEAD=kv_pad_head,
    )
    return o_h.view(B, N, C), lse


def _flash_topk_attn_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    merged_indices: torch.Tensor,
    qblock_cu_seqlens: torch.Tensor,
    num_heads: int,
    q_block_size: int,
    kv_block_size: int,
    scale: Optional[float] = None,
    kv_padding: Tuple[int, int] = (0, 0),
    q_padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference: same semantics as Triton (union over q-block, softmax over union tokens)."""
    dtype = q.dtype
    B, N, C = q.shape
    H = num_heads
    D = C // H
    Q_BS, KV_BS = q_block_size, kv_block_size

    assert k.shape == (B, N, C) and v.shape == (B, N, C)
    assert C == H * D

    kv_pad_head, kv_pad_tail = kv_padding
    q_pad_head, q_pad_tail = q_padding
    N_Q_PADDED = q_pad_head + N + q_pad_tail
    QM = N_Q_PADDED // Q_BS

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_f = rearrange(q, "b n (h d) -> b h n d", h=H).float()
    k_f = rearrange(k, "b n (h d) -> b h n d", h=H).float()
    v_f = rearrange(v, "b n (h d) -> b h n d", h=H).float()

    # H_BI is implicit in merged_indices/qblock_cu_seqlens; enforce in {1, H}.
    H_BI = merged_indices.shape[1]
    assert H_BI == 1 or H_BI == H

    device = q.device

    o = torch.zeros(B, H, N, D, device=device, dtype=torch.float32)
    lse_out = torch.full((B, H, N), float("-inf"), device=device, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for qm in range(QM):
                # Real query range for this q-block.
                q_start = max(0, qm * Q_BS - q_pad_head)
                q_end = min(N, (qm + 1) * Q_BS - q_pad_head)
                if q_start >= q_end:
                    continue

                start = qblock_cu_seqlens[b, h, qm].item()
                end = qblock_cu_seqlens[b, h, qm + 1].item()
                blocks = merged_indices[b, h, start:end]
                blocks = blocks[blocks >= 0]

                valid = torch.zeros(N, dtype=torch.bool, device=device)
                for bid in blocks.tolist():
                    t_start = max(0, bid * KV_BS - kv_pad_head)
                    t_end = min(N, (bid + 1) * KV_BS - kv_pad_head)
                    if t_start < t_end:
                        valid[t_start:t_end] = True

                has_any = valid.any()
                if not has_any:
                    continue

                for n in range(q_start, q_end):
                    s = torch.einsum("d,dn->n", q_f[b, h, n] * scale, k_f[b, h].transpose(0, 1))
                    s = s.masked_fill(~valid, float("-inf"))
                    lse = torch.logsumexp(s, dim=-1)
                    p = torch.exp(s - lse)
                    o[b, h, n] = torch.einsum("n,nd->d", p, v_f[b, h])
                    lse_out[b, h, n] = lse

    o = rearrange(o, "b h n d -> b n (h d)")
    return o.to(dtype), lse_out
