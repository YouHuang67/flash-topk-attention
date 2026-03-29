"""
Q-block shared KV-candidate sparse attention: exact union of per-query top-k blocks
within each q_block_size group; all queries in the block attend over the same merged set.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

from flash_topk_attn.scoring import _next_power_of_2


def build_qblock_merged_indices(topk_indices: torch.Tensor, Q_BS: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sorted unique block ids per q-block (exact union).

    Args:
        topk_indices: [B, H, N, KQ] int32 (or int64); -1 marks empty slots.
        Q_BS: query block size; N must be divisible by Q_BS.

    Returns:
        merged_indices: [B, H, N * KQ] int32, valid prefix then -1 padding.
        qblock_cu_seqlens: [B, H, QM + 1] int32 cumulative lengths per q-block.
    """
    B, H, N, KQ = topk_indices.shape
    assert N % Q_BS == 0, f"N={N} must be divisible by Q_BS={Q_BS}"
    QM = N // Q_BS
    device = topk_indices.device

    ids = topk_indices.to(torch.int32).view(B, H, QM, Q_BS * KQ)
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
):
    """One KV sub-step: load K/V, tl.dot scores, online softmax in log2 domain."""
    if BRANCH_ID == 1:
        # KV_TILE >= KV_BS: semantic block (KV_BS) with physical tile KV_TILE (2^k).
        # base absorbs token_offset; shape caps logical tensor; tl.dot uses KV_TILE width.
        token_offset = kv_block * KV_BS
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
        token_offset = kv_block * KV_BS + kv_sub_iter * KV_TILE
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
        in_block = kv_idx < kv_block * KV_BS + KV_BS
        if BRANCH_ID == 3:
            tile_valid = (kv_idx < N) & in_block & (tl.arange(0, KV_TILE) < TAIL_LEN)
        else:
            tile_valid = (kv_idx < N) & in_block

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
    q_subtile_offset = qm * Q_BS + q_tile_i * Q_TILE

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
    if Q_BS % Q_TILE == 0 and D % BD == 0:
        b_q = tl.load(p_q).to(tl.float32) * scale_log2
    elif Q_BS % Q_TILE == 0:
        b_q = tl.load(p_q, boundary_check=(1,)).to(tl.float32) * scale_log2
    elif D % BD == 0:
        b_q = tl.load(p_q, boundary_check=(0,)).to(tl.float32) * scale_log2
    else:
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * scale_log2

    q_row = q_subtile_offset + tl.arange(0, Q_TILE)
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
            )
        else:
            if USE_STATIC_INNER:
                for kv_sub_iter in tl.static_range(INNER_FULL):
                    b_m, b_l, b_o = _qblock_accum_kv(
                        kv_block, kv_sub_iter, k_base, v_base, b_q,
                        stride_k_d, stride_k_t, stride_v_t, stride_v_d, i_v, N, b_m, b_l, b_o,
                        KV_BS=KV_BS, KV_TILE=KV_TILE, BD=BD, BV=BV, D=D,
                        BRANCH_ID=2, TAIL_LEN=0, NEG_INF_C=NEG_INF_C,
                    )
            else:
                for kv_sub_iter in range(INNER_FULL):
                    b_m, b_l, b_o = _qblock_accum_kv(
                        kv_block, kv_sub_iter, k_base, v_base, b_q,
                        stride_k_d, stride_k_t, stride_v_t, stride_v_d, i_v, N, b_m, b_l, b_o,
                        KV_BS=KV_BS, KV_TILE=KV_TILE, BD=BD, BV=BV, D=D,
                        BRANCH_ID=2, TAIL_LEN=0, NEG_INF_C=NEG_INF_C,
                    )
            if INNER_TAIL:
                b_m, b_l, b_o = _qblock_accum_kv(
                    kv_block, INNER_FULL, k_base, v_base, b_q,
                    stride_k_d, stride_k_t, stride_v_t, stride_v_d, i_v, N, b_m, b_l, b_o,
                    KV_BS=KV_BS, KV_TILE=KV_TILE, BD=BD, BV=BV, D=D,
                    BRANCH_ID=3, TAIL_LEN=TAIL_LEN, NEG_INF_C=NEG_INF_C,
                )

    safe_l = tl.where(b_l > 0.0, b_l, 1.0)
    final_o = b_o / safe_l[:, None]

    o_row = q_subtile_offset + tl.arange(0, Q_TILE)
    v_c = i_v * BV + tl.arange(0, BV)
    o_base = o_ptr + i_b * stride_o_b + i_h * stride_o_h
    o_offs = o_row[:, None] * stride_o_t + v_c[None, :] * stride_o_d
    tl.store(
        o_base + o_offs,
        final_o.to(o_ptr.dtype.element_ty),
        mask=q_valid[:, None] & (v_c[None, :] < D),
    )

    if i_v == 0:
        LN2_C: tl.constexpr = 0.6931471805599453
        final_lse = (b_m + tl.log2(safe_l)) * LN2_C
        lse_base = lse_ptr + i_b * stride_lse_b + i_h * stride_lse_h
        tl.store(
            lse_base + q_row * stride_lse_t,
            final_lse.to(lse_ptr.dtype.element_ty),
            mask=q_valid,
        )


def flash_topk_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk_indices: torch.Tensor,
    num_heads: int, q_block_size: int, kv_block_size: int,
    scale: Optional[float] = None, num_kv_heads: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention with exact per-q-block union of top-k KV blocks.

    Args:
        q, k, v: [B, N, C], C = num_heads * D.
        topk_indices: [B, H_BI, N, KQ] int32; each value is a KV block id.
        num_heads: H.
        q_block_size: queries per shared-candidate group; N % q_block_size == 0.
        kv_block_size: tokens per block id; must match scoring; N % kv_block_size == 0.
        scale: defaults to 1/sqrt(D).
        num_kv_heads: v1 requires num_kv_heads == num_heads.

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
    assert N % Q_BS == 0 and N % KV_BS == 0

    if num_kv_heads is not None and num_kv_heads != H:
        raise NotImplementedError("GQA (num_kv_heads != num_heads) is not supported in v1")

    H_BI = topk_indices.shape[1]
    assert H_BI == 1 or H_BI == H

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    merged_indices, qblock_cu_seqlens = build_qblock_merged_indices(
        topk_indices.contiguous(), Q_BS
    )

    q_h = q.contiguous().view(B, N, H, D)
    k_h = k.contiguous().view(B, N, H, D)
    v_h = v.contiguous().view(B, N, H, D)
    o_h = torch.empty_like(q_h)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

    BD = min(128, _next_power_of_2(D))
    BV = min(BD, 64)
    QM = N // Q_BS

    grid = lambda meta: (
        QM * triton.cdiv(Q_BS, meta["Q_TILE"]),
        triton.cdiv(D, BV),
        B * H,
    )

    _flash_topk_attn_fwd_kernel[grid](
        q_h, k_h, v_h, merged_indices, qblock_cu_seqlens, o_h, lse,
        *q_h.stride(), *k_h.stride(), *v_h.stride(),
        *merged_indices.stride(), *qblock_cu_seqlens.stride(), *o_h.stride(),
        lse.stride(0), lse.stride(1), lse.stride(2),
        B, H, H_BI, N, D, QM, scale,
        Q_BS=Q_BS, KV_BS=KV_BS, BD=BD, BV=BV,
    )
    return o_h.view(B, N, C), lse


def _flash_topk_attn_naive(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk_indices: torch.Tensor,
    num_heads: int, q_block_size: int, kv_block_size: int, scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference: same semantics as Triton (union over q-block, softmax over union tokens)."""
    dtype = q.dtype
    B, N, C = q.shape
    H = num_heads
    D = C // H
    Q_BS, KV_BS = q_block_size, kv_block_size
    KQ = topk_indices.shape[-1]

    assert k.shape == (B, N, C) and v.shape == (B, N, C)
    assert C == H * D
    assert N % Q_BS == 0 and N % KV_BS == 0
    H_BI = topk_indices.shape[1]
    assert H_BI == 1 or H_BI == H
    assert topk_indices.shape == (B, H_BI, N, KQ)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_f = rearrange(q, "b n (h d) -> b h n d", h=H).float()
    k_f = rearrange(k, "b n (h d) -> b h n d", h=H).float()
    v_f = rearrange(v, "b n (h d) -> b h n d", h=H).float()

    bi = topk_indices.long()
    if bi.shape[1] == 1 and H > 1:
        bi = bi.expand(B, H, N, KQ)

    QM = N // Q_BS
    device = q.device
    kv_block_id = torch.arange(N, device=device, dtype=torch.long) // KV_BS

    o = torch.zeros(B, H, N, D, device=device, dtype=torch.float32)
    lse_out = torch.full((B, H, N), float("-inf"), device=device, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for qm in range(QM):
                n0 = qm * Q_BS
                n1 = n0 + Q_BS
                slab = bi[b, h, n0:n1, :].reshape(-1)
                pos = slab[slab >= 0]
                if pos.numel() == 0:
                    blocks = torch.empty(0, device=device, dtype=torch.long)
                else:
                    blocks = torch.unique(pos)
                    blocks, _ = torch.sort(blocks)

                if blocks.numel() == 0:
                    valid = torch.zeros(N, dtype=torch.bool, device=device)
                else:
                    valid = torch.isin(kv_block_id, blocks)

                has_any = valid.any()
                for n in range(n0, n1):
                    if not has_any:
                        continue
                    s = torch.einsum("d,dn->n", q_f[b, h, n] * scale, k_f[b, h].transpose(0, 1))
                    s = s.masked_fill(~valid, float("-inf"))
                    lse = torch.logsumexp(s, dim=-1)
                    p = torch.exp(s - lse)
                    o[b, h, n] = torch.einsum("n,nd->d", p, v_f[b, h])
                    lse_out[b, h, n] = lse

    o = rearrange(o, "b h n d -> b n (h d)")
    return o.to(dtype), lse_out
