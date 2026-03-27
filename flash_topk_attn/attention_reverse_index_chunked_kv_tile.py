"""Reverse-index chunked sparse top-k block attention (KV sub-tiling variant)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flash_topk_attn.scoring import _next_power_of_2

NEG_INF = -1e38
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def build_reverse_index(
    block_indices: torch.Tensor, M: int, topk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort flat indices by block id and return offsets."""
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
def _load_q_batch(
    sf_ptr,
    q_base,
    start,
    q_off,
    query_count,
    topk,
    scale_log2,
    stride_q_t,
    stride_q_d,
    D,
    Q_BS: tl.constexpr,
    BD: tl.constexpr,
):
    actual = tl.minimum(Q_BS, query_count - q_off)
    q_mask = tl.arange(0, Q_BS) < actual

    flat_idx = tl.load(
        sf_ptr + start + q_off + tl.arange(0, Q_BS),
        mask=q_mask,
        other=0,
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
    return b_q, q_mask, q_pos, slot


@triton.jit
def _store_mid(
    mo_base,
    ml_base,
    b_o,
    b_lse,
    q_pos,
    slot,
    q_mask,
    i_v,
    stride_mo_t,
    stride_mo_k,
    stride_mo_d,
    stride_ml_t,
    stride_ml_k,
    D,
    BV: tl.constexpr,
):
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
def _accum_kv_tile(
    tile_i,
    k_base,
    v_base,
    b_q,
    q_mask,
    stride_k_d,
    stride_k_t,
    stride_v_t,
    stride_v_d,
    kv_offset,
    i_v,
    N,
    D,
    b_m,
    b_l,
    b_o,
    NEG_INF_C: tl.constexpr,
    BD: tl.constexpr,
    KV_TILE: tl.constexpr,
    BV: tl.constexpr,
):
    tile_offset = kv_offset + tile_i * KV_TILE
    p_k = tl.make_block_ptr(
        base=k_base,
        shape=(D, N),
        strides=(stride_k_d, stride_k_t),
        offsets=(0, tile_offset),
        block_shape=(BD, KV_TILE),
        order=(0, 1),
    )
    p_v = tl.make_block_ptr(
        base=v_base,
        shape=(N, D),
        strides=(stride_v_t, stride_v_d),
        offsets=(tile_offset, i_v * BV),
        block_shape=(KV_TILE, BV),
        order=(1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0,)).to(tl.float32)

    b_s = tl.dot(b_q, b_k)
    b_s = tl.where(q_mask[:, None], b_s, NEG_INF_C)

    b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
    scale_old = tl.exp2(b_m - b_m_new)
    b_p = tl.exp2(b_s - b_m_new[:, None])

    b_l = b_l * scale_old + tl.sum(b_p, axis=1)
    b_o = b_o * scale_old[:, None] + tl.dot(b_p, b_v)
    b_m = b_m_new
    return b_m, b_l, b_o


@triton.jit
def _accum_kv_tile_tail(
    tile_i,
    k_base,
    v_base,
    b_q,
    q_mask,
    stride_k_d,
    stride_k_t,
    stride_v_t,
    stride_v_d,
    kv_offset,
    i_v,
    N,
    D,
    b_m,
    b_l,
    b_o,
    NEG_INF_C: tl.constexpr,
    TAIL_LEN: tl.constexpr,
    BD: tl.constexpr,
    KV_TILE: tl.constexpr,
    BV: tl.constexpr,
):
    tile_offset = kv_offset + tile_i * KV_TILE
    p_k = tl.make_block_ptr(
        base=k_base,
        shape=(D, N),
        strides=(stride_k_d, stride_k_t),
        offsets=(0, tile_offset),
        block_shape=(BD, KV_TILE),
        order=(0, 1),
    )
    p_v = tl.make_block_ptr(
        base=v_base,
        shape=(N, D),
        strides=(stride_v_t, stride_v_d),
        offsets=(tile_offset, i_v * BV),
        block_shape=(KV_TILE, BV),
        order=(1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

    b_s = tl.dot(b_q, b_k)
    tile_valid = tl.arange(0, KV_TILE) < TAIL_LEN
    b_s = tl.where(q_mask[:, None] & tile_valid[None, :], b_s, NEG_INF_C)

    b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
    scale_old = tl.exp2(b_m - b_m_new)
    b_p = tl.exp2(b_s - b_m_new[:, None])

    b_l = b_l * scale_old + tl.sum(b_p, axis=1)
    b_o = b_o * scale_old[:, None] + tl.dot(b_p, b_v)
    b_m = b_m_new
    return b_m, b_l, b_o


@triton.autotune(
    configs=[
        triton.Config({"KV_TILE": kv}, num_warps=w)
        for kv in (16, 32, 64)
        for w in (2, 4)
    ],
    key=["N", "SCORE_BS_ORIG", "D"],
)
@triton.jit
def _reverse_attn_stage1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    sorted_flat_idx_ptr,
    block_offsets_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_t,
    stride_q_d,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_sf_b,
    stride_sf_h,
    stride_bo_b,
    stride_bo_h,
    stride_mo_b,
    stride_mo_h,
    stride_mo_t,
    stride_mo_k,
    stride_mo_d,
    stride_ml_b,
    stride_ml_h,
    stride_ml_t,
    stride_ml_k,
    B,
    H,
    H_BI,
    M,
    N,
    D,
    topk,
    scale,
    SCORE_BS_ORIG: tl.constexpr,
    Q_BS: tl.constexpr,
    KV_TILE: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
):
    NEG_INF_C: tl.constexpr = -1e38
    LOG2E_C: tl.constexpr = 1.4426950408889634
    NUM_KV_TILES: tl.constexpr = (SCORE_BS_ORIG + KV_TILE - 1) // KV_TILE
    IS_EXACT: tl.constexpr = SCORE_BS_ORIG % KV_TILE == 0
    FULL_TILES: tl.constexpr = SCORE_BS_ORIG // KV_TILE
    HAS_TAIL: tl.constexpr = SCORE_BS_ORIG % KV_TILE != 0
    TAIL_LEN: tl.constexpr = SCORE_BS_ORIG - FULL_TILES * KV_TILE
    USE_STATIC: tl.constexpr = NUM_KV_TILES <= 16

    i_m = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = (i_bh // H).to(tl.int64)
    i_h = (i_bh % H).to(tl.int64)
    i_h_bi = tl.minimum(i_h, H_BI - 1)

    scale_log2 = scale * LOG2E_C

    bo_ptr = block_offsets_ptr + i_b * stride_bo_b + i_h_bi * stride_bo_h
    start = tl.load(bo_ptr + i_m)
    end = tl.load(bo_ptr + i_m + 1)
    query_count = end - start
    if query_count == 0:
        return

    kv_offset = i_m * SCORE_BS_ORIG
    sf_ptr = sorted_flat_idx_ptr + i_b * stride_sf_b + i_h_bi * stride_sf_h
    q_base = q_ptr + i_b * stride_q_b + i_h * stride_q_h
    k_base = k_ptr + i_b * stride_k_b + i_h * stride_k_h
    v_base = v_ptr + i_b * stride_v_b + i_h * stride_v_h
    mo_base = mid_o_ptr + i_b * stride_mo_b + i_h * stride_mo_h
    ml_base = mid_lse_ptr + i_b * stride_ml_b + i_h * stride_ml_h

    if NUM_KV_TILES == 1:
        p_k = tl.make_block_ptr(
            base=k_base,
            shape=(D, N),
            strides=(stride_k_d, stride_k_t),
            offsets=(0, kv_offset),
            block_shape=(BD, KV_TILE),
            order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            base=v_base,
            shape=(N, D),
            strides=(stride_v_t, stride_v_d),
            offsets=(kv_offset, i_v * BV),
            block_shape=(KV_TILE, BV),
            order=(1, 0),
        )

        if IS_EXACT:
            b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
            b_v = tl.load(p_v, boundary_check=(0,)).to(tl.float32)
            for q_off in range(0, query_count, Q_BS):
                b_q, q_mask, q_pos, slot = _load_q_batch(
                    sf_ptr,
                    q_base,
                    start,
                    q_off,
                    query_count,
                    topk,
                    scale_log2,
                    stride_q_t,
                    stride_q_d,
                    D,
                    Q_BS=Q_BS,
                    BD=BD,
                )
                b_s = tl.dot(b_q, b_k)
                b_s = tl.where(q_mask[:, None], b_s, NEG_INF_C)
                b_m = tl.max(b_s, axis=1)
                b_p = tl.exp2(b_s - b_m[:, None])
                b_l = tl.sum(b_p, axis=1)
                b_o = tl.dot(b_p, b_v) / b_l[:, None]
                b_lse = b_m + tl.log2(b_l)
                _store_mid(
                    mo_base,
                    ml_base,
                    b_o,
                    b_lse,
                    q_pos,
                    slot,
                    q_mask,
                    i_v,
                    stride_mo_t,
                    stride_mo_k,
                    stride_mo_d,
                    stride_ml_t,
                    stride_ml_k,
                    D,
                    BV=BV,
                )
        else:
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
            kv_valid = tl.arange(0, KV_TILE) < SCORE_BS_ORIG
            for q_off in range(0, query_count, Q_BS):
                b_q, q_mask, q_pos, slot = _load_q_batch(
                    sf_ptr,
                    q_base,
                    start,
                    q_off,
                    query_count,
                    topk,
                    scale_log2,
                    stride_q_t,
                    stride_q_d,
                    D,
                    Q_BS=Q_BS,
                    BD=BD,
                )
                b_s = tl.dot(b_q, b_k)
                b_s = tl.where(q_mask[:, None] & kv_valid[None, :], b_s, NEG_INF_C)
                b_m = tl.max(b_s, axis=1)
                b_p = tl.exp2(b_s - b_m[:, None])
                b_l = tl.sum(b_p, axis=1)
                b_o = tl.dot(b_p, b_v) / b_l[:, None]
                b_lse = b_m + tl.log2(b_l)
                _store_mid(
                    mo_base,
                    ml_base,
                    b_o,
                    b_lse,
                    q_pos,
                    slot,
                    q_mask,
                    i_v,
                    stride_mo_t,
                    stride_mo_k,
                    stride_mo_d,
                    stride_ml_t,
                    stride_ml_k,
                    D,
                    BV=BV,
                )
        return

    for q_off in range(0, query_count, Q_BS):
        b_q, q_mask, q_pos, slot = _load_q_batch(
            sf_ptr,
            q_base,
            start,
            q_off,
            query_count,
            topk,
            scale_log2,
            stride_q_t,
            stride_q_d,
            D,
            Q_BS=Q_BS,
            BD=BD,
        )
        b_m = tl.full([Q_BS], float("-inf"), dtype=tl.float32)
        b_l = tl.zeros([Q_BS], dtype=tl.float32)
        b_o = tl.zeros([Q_BS, BV], dtype=tl.float32)

        if USE_STATIC:
            for tile_i in tl.static_range(FULL_TILES):
                b_m, b_l, b_o = _accum_kv_tile(
                    tile_i,
                    k_base,
                    v_base,
                    b_q,
                    q_mask,
                    stride_k_d,
                    stride_k_t,
                    stride_v_t,
                    stride_v_d,
                    kv_offset,
                    i_v,
                    N,
                    D,
                    b_m,
                    b_l,
                    b_o,
                    NEG_INF_C=NEG_INF_C,
                    BD=BD,
                    KV_TILE=KV_TILE,
                    BV=BV,
                )
        else:
            for tile_i in range(FULL_TILES):
                b_m, b_l, b_o = _accum_kv_tile(
                    tile_i,
                    k_base,
                    v_base,
                    b_q,
                    q_mask,
                    stride_k_d,
                    stride_k_t,
                    stride_v_t,
                    stride_v_d,
                    kv_offset,
                    i_v,
                    N,
                    D,
                    b_m,
                    b_l,
                    b_o,
                    NEG_INF_C=NEG_INF_C,
                    BD=BD,
                    KV_TILE=KV_TILE,
                    BV=BV,
                )

        if HAS_TAIL:
            b_m, b_l, b_o = _accum_kv_tile_tail(
                FULL_TILES,
                k_base,
                v_base,
                b_q,
                q_mask,
                stride_k_d,
                stride_k_t,
                stride_v_t,
                stride_v_d,
                kv_offset,
                i_v,
                N,
                D,
                b_m,
                b_l,
                b_o,
                NEG_INF_C=NEG_INF_C,
                TAIL_LEN=TAIL_LEN,
                BD=BD,
                KV_TILE=KV_TILE,
                BV=BV,
            )

        b_o = b_o / b_l[:, None]
        b_lse = b_m + tl.log2(b_l)
        _store_mid(
            mo_base,
            ml_base,
            b_o,
            b_lse,
            q_pos,
            slot,
            q_mask,
            i_v,
            stride_mo_t,
            stride_mo_k,
            stride_mo_d,
            stride_ml_t,
            stride_ml_k,
            D,
            BV=BV,
        )


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
    LN2_C: tl.constexpr = 0.6931471805599453

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
        final_lse = (acc_m + tl.log2(acc_l)) * LN2_C
        tl.store(
            lse_ptr + i_b * stride_lse_b + i_h * stride_lse_h + i_t * stride_lse_t,
            final_lse,
        )


@triton.jit
def _chunked_merge_kernel(
    mid_o_ptr, mid_lse_ptr,
    acc_o_ptr, acc_m_ptr, acc_l_ptr,
    stride_mo_b, stride_mo_h, stride_mo_t, stride_mo_k, stride_mo_d,
    stride_ml_b, stride_ml_h, stride_ml_t, stride_ml_k,
    stride_ao_b, stride_ao_h, stride_ao_t, stride_ao_d,
    stride_am_b, stride_am_h, stride_am_t,
    B, H, N, D, chunk_k,
    BV: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = (i_bh // H).to(tl.int64)
    i_h = (i_bh % H).to(tl.int64)

    am_off = i_b * stride_am_b + i_h * stride_am_h + i_t * stride_am_t
    acc_m = tl.load(acc_m_ptr + am_off)
    acc_l = tl.load(acc_l_ptr + am_off)

    ao_base = acc_o_ptr + i_b * stride_ao_b + i_h * stride_ao_h + i_t * stride_ao_t
    p_ao = tl.make_block_ptr(
        base=ao_base,
        shape=(D,),
        strides=(stride_ao_d,),
        offsets=(i_v * BV,),
        block_shape=(BV,),
        order=(0,),
    )
    acc_o = tl.load(p_ao, boundary_check=(0,)).to(tl.float32)

    mo_base = mid_o_ptr + i_b * stride_mo_b + i_h * stride_mo_h + i_t * stride_mo_t
    ml_base = mid_lse_ptr + i_b * stride_ml_b + i_h * stride_ml_h + i_t * stride_ml_t

    for i_k in range(chunk_k):
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

    tl.store(p_ao, acc_o, boundary_check=(0,))

    if i_v == 0:
        tl.store(acc_m_ptr + am_off, acc_m)
        tl.store(acc_l_ptr + am_off, acc_l)


@triton.jit
def _chunked_finalize_kernel(
    acc_o_ptr, acc_m_ptr, acc_l_ptr,
    o_ptr, lse_ptr,
    stride_ao_b, stride_ao_h, stride_ao_t, stride_ao_d,
    stride_am_b, stride_am_h, stride_am_t,
    stride_o_b, stride_o_t, stride_o_h, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_t,
    B, H, N, D,
    BV: tl.constexpr,
):
    LN2_C: tl.constexpr = 0.6931471805599453

    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = (i_bh // H).to(tl.int64)
    i_h = (i_bh % H).to(tl.int64)

    ao_base = acc_o_ptr + i_b * stride_ao_b + i_h * stride_ao_h + i_t * stride_ao_t
    p_ao = tl.make_block_ptr(
        base=ao_base,
        shape=(D,),
        strides=(stride_ao_d,),
        offsets=(i_v * BV,),
        block_shape=(BV,),
        order=(0,),
    )
    acc_o = tl.load(p_ao, boundary_check=(0,)).to(tl.float32)

    am_off = i_b * stride_am_b + i_h * stride_am_h + i_t * stride_am_t
    acc_l = tl.load(acc_l_ptr + am_off)

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
        acc_m = tl.load(acc_m_ptr + am_off)
        final_lse = (acc_m + tl.log2(acc_l)) * LN2_C
        tl.store(
            lse_ptr + i_b * stride_lse_b + i_h * stride_lse_h + i_t * stride_lse_t,
            final_lse,
        )


def _auto_chunk_size(
    B: int, H: int, N: int, topk: int, D: int,
    dtype: torch.dtype = torch.float16,
    max_mid_bytes: int = 2 * 1024**3,
) -> int:
    """Choose the largest chunk size that fits the configured memory budget.

    Accounted buffers:
        mid_o:   B * H * N * C * D * dtype_size
        mid_lse: B * H * N * C * 4
        acc_o:   B * H * N * D * 4
        acc_m:   B * H * N * 4
        acc_l:   B * H * N * 4
    """
    bhn = B * H * N
    dtype_size = torch.finfo(dtype).bits // 8
    acc_bytes = bhn * (D + 2) * 4
    available = max(0, max_mid_bytes - acc_bytes)
    per_slot = bhn * (D * dtype_size + 4)
    c = max(1, available // per_slot) if per_slot > 0 else topk
    return min(c, topk)


def flash_topk_attn_reverse_index_chunked_kv_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    num_heads: int,
    block_size: int = 64,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    q_bs: int = 16,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Sparse attention with chunked reverse-index and KV sub-tiling in stage1."""
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
    SCORE_BS_ORIG = block_size

    if chunk_size <= 0:
        chunk_size = _auto_chunk_size(B, H, N, topk, D, dtype=q.dtype)
    chunk_size = min(chunk_size, topk)
    chunk_size = max(1, chunk_size)

    q_h = q.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    k_h = k.view(B, N, H, D)
    v_h = v.view(B, N, H, D)
    o_h = torch.empty(B, N, H, D, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

    if chunk_size >= topk:
        sorted_flat_idx, block_offsets = build_reverse_index(block_indices, M, topk)
        mid_o = torch.zeros(B, H, N, topk, D, device=q.device, dtype=q.dtype)
        mid_lse = torch.full((B, H, N, topk), NEG_INF, device=q.device, dtype=torch.float32)

        grid_stage1 = (M, triton.cdiv(D, BV), B * H)
        _reverse_attn_stage1_kernel[grid_stage1](
            q_h,
            k_h,
            v_h,
            sorted_flat_idx,
            block_offsets,
            mid_o,
            mid_lse,
            *q_h.stride(),
            *k_h.stride(),
            *v_h.stride(),
            sorted_flat_idx.stride(0),
            sorted_flat_idx.stride(1),
            block_offsets.stride(0),
            block_offsets.stride(1),
            *mid_o.stride(),
            *mid_lse.stride(),
            B=B,
            H=H,
            H_BI=H_BI,
            M=M,
            N=N,
            D=D,
            topk=topk,
            scale=scale,
            SCORE_BS_ORIG=SCORE_BS_ORIG,
            Q_BS=Q_BS,
            BD=BD,
            BV=BV,
        )

        grid_stage2 = (N, triton.cdiv(D, BV), B * H)
        _reverse_attn_stage2_kernel[grid_stage2](
            mid_o, mid_lse, o_h, lse,
            *mid_o.stride(),
            *mid_lse.stride(),
            *o_h.stride(),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B=B, H=H, N=N, D=D, topk=topk,
            BV=BV,
        )
        return o_h.view(B, N, C)

    num_chunks = (topk + chunk_size - 1) // chunk_size
    mid_o = torch.empty(B, H, N, chunk_size, D, device=q.device, dtype=q.dtype)
    mid_lse = torch.empty(B, H, N, chunk_size, device=q.device, dtype=torch.float32)
    acc_o = torch.zeros(B, H, N, D, device=q.device, dtype=torch.float32)
    acc_m = torch.full((B, H, N), NEG_INF, device=q.device, dtype=torch.float32)
    acc_l = torch.zeros(B, H, N, device=q.device, dtype=torch.float32)

    grid_stage1 = (M, triton.cdiv(D, BV), B * H)
    grid_merge = (N, triton.cdiv(D, BV), B * H)

    for i_chunk in range(num_chunks):
        mid_o.zero_()
        mid_lse.fill_(NEG_INF)

        start = i_chunk * chunk_size
        end = min(start + chunk_size, topk)
        bi_chunk = block_indices[:, :, :, start:end]
        if end - start < chunk_size:
            bi_chunk = F.pad(bi_chunk, (0, chunk_size - (end - start)), value=-1)
        bi_chunk = bi_chunk.contiguous()

        sf_chunk, bo_chunk = build_reverse_index(bi_chunk, M, chunk_size)
        _reverse_attn_stage1_kernel[grid_stage1](
            q_h,
            k_h,
            v_h,
            sf_chunk,
            bo_chunk,
            mid_o,
            mid_lse,
            *q_h.stride(),
            *k_h.stride(),
            *v_h.stride(),
            sf_chunk.stride(0),
            sf_chunk.stride(1),
            bo_chunk.stride(0),
            bo_chunk.stride(1),
            *mid_o.stride(),
            *mid_lse.stride(),
            B=B,
            H=H,
            H_BI=H_BI,
            M=M,
            N=N,
            D=D,
            topk=chunk_size,
            scale=scale,
            SCORE_BS_ORIG=SCORE_BS_ORIG,
            Q_BS=Q_BS,
            BD=BD,
            BV=BV,
        )

        _chunked_merge_kernel[grid_merge](
            mid_o, mid_lse,
            acc_o, acc_m, acc_l,
            *mid_o.stride(), *mid_lse.stride(),
            acc_o.stride(0), acc_o.stride(1), acc_o.stride(2), acc_o.stride(3),
            acc_m.stride(0), acc_m.stride(1), acc_m.stride(2),
            B=B, H=H, N=N, D=D, chunk_k=chunk_size, BV=BV,
        )

    _chunked_finalize_kernel[grid_merge](
        acc_o,
        acc_m,
        acc_l,
        o_h,
        lse,
        acc_o.stride(0),
        acc_o.stride(1),
        acc_o.stride(2),
        acc_o.stride(3),
        acc_m.stride(0),
        acc_m.stride(1),
        acc_m.stride(2),
        *o_h.stride(),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        B=B,
        H=H,
        N=N,
        D=D,
        BV=BV,
    )
    return o_h.view(B, N, C)

