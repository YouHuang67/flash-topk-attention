"""
Sparse attention over top-k KV blocks (forward only in v0; backward placeholder).
"""
import math
from typing import Optional

import torch
import triton
import triton.language as tl
from einops import rearrange

from flash_topk_attn.scoring import _next_power_of_2


@triton.jit
def _flash_topk_attn_fwd_accumulate_sub(
    kv_sub_iter, score_block_id,
    k_base, v_base,
    b_q,
    stride_k_d, stride_k_t, stride_v_t, stride_v_d,
    i_v,
    BV: tl.constexpr, D: tl.constexpr, N: tl.constexpr, BD: tl.constexpr,
    KV_BS: tl.constexpr, SCORE_BS_ORIG: tl.constexpr, SCORE_BS: tl.constexpr,
    BRANCH_ID: tl.constexpr, NEG_INF: tl.constexpr,
    b_m, b_acc, b_o,
):
    """One inner sub-step: load K/V tile, scores, online softmax update."""
    # BRANCH_ID 1: IS_POW2 and KV_BS >= SCORE_BS (former 0/1 merged).
    if BRANCH_ID == 1:
        token_offset = score_block_id * SCORE_BS

        p_k = tl.make_block_ptr(
            base=k_base,
            shape=(D, N),
            strides=(stride_k_d, stride_k_t),
            offsets=(0, token_offset),
            block_shape=(BD, SCORE_BS),
            order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            base=v_base,
            shape=(N, D),
            strides=(stride_v_t, stride_v_d),
            offsets=(token_offset, i_v * BV),
            block_shape=(SCORE_BS, BV),
            order=(1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

        b_s = tl.sum(b_q[:, None] * b_k, axis=0)

    # BRANCH_ID 2: IS_POW2 and KV_BS < SCORE_BS (one score block spans multiple KV tiles via kv_sub_iter).
    elif BRANCH_ID == 2:
        block_base = score_block_id * SCORE_BS_ORIG
        token_offset = block_base + kv_sub_iter * KV_BS

        p_k = tl.make_block_ptr(
            base=k_base,
            shape=(D, N),
            strides=(stride_k_d, stride_k_t),
            offsets=(0, token_offset),
            block_shape=(BD, KV_BS),
            order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            base=v_base,
            shape=(N, D),
            strides=(stride_v_t, stride_v_d),
            offsets=(token_offset, i_v * BV),
            block_shape=(KV_BS, BV),
            order=(1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

        b_s = tl.sum(b_q[:, None] * b_k, axis=0)

    # BRANCH_ID 3: not IS_POW2 (former 3/4/5 merged).
    else:
        block_base = score_block_id * SCORE_BS_ORIG
        offset_in_block = kv_sub_iter * KV_BS

        p_k = tl.make_block_ptr(
            base=k_base + block_base * stride_k_t,
            shape=(D, SCORE_BS_ORIG),
            strides=(stride_k_d, stride_k_t),
            offsets=(0, offset_in_block),
            block_shape=(BD, KV_BS),
            order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            base=v_base + block_base * stride_v_t,
            shape=(SCORE_BS_ORIG, D),
            strides=(stride_v_t, stride_v_d),
            offsets=(offset_in_block, i_v * BV),
            block_shape=(KV_BS, BV),
            order=(1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)

        b_s = tl.sum(b_q[:, None] * b_k, axis=0)
        kv_idx = tl.arange(0, KV_BS)
        valid = (offset_in_block + kv_idx) < SCORE_BS_ORIG
        b_s = tl.where(valid, b_s, NEG_INF)

    b_m_new = tl.maximum(b_m, tl.max(b_s, 0))
    b_r = tl.exp(b_m - b_m_new)
    b_p = tl.exp(b_s - b_m_new)
    b_acc = b_acc * b_r + tl.sum(b_p, 0)
    b_o = b_o * b_r + tl.sum(b_p[:, None] * b_v, axis=0)
    b_m = b_m_new
    return b_m, b_acc, b_o


@triton.autotune(
    configs=[
        triton.Config({"KV_BS": kv}, num_warps=w) for kv in (32, 64) for w in (2, 4, 8)
    ],
    key=["N", "SCORE_BS_ORIG", "SCORE_BS", "D"],
)
@triton.jit
def _flash_topk_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, bi_ptr,
    stride_q_b, stride_q_t, stride_q_h, stride_q_d,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_o_b, stride_o_t, stride_o_h, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_n,
    stride_bi_b, stride_bi_h, stride_bi_n, stride_bi_k,
    N, H, H_BI,
    D: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
    SCORE_BS_ORIG: tl.constexpr, SCORE_BS: tl.constexpr, KV_BS: tl.constexpr,
    scale, topk,
):
    """Forward kernel: one query token, one value tile; BRANCH_ID / INNER_ITERS derived inside."""
    NEG_INF: tl.constexpr = -1e38

    IS_POW2: tl.constexpr = SCORE_BS_ORIG == SCORE_BS
    if IS_POW2:
        if KV_BS >= SCORE_BS:
            BRANCH_ID: tl.constexpr = 1
            INNER_ITERS: tl.constexpr = 1
        else:
            BRANCH_ID: tl.constexpr = 2
            INNER_ITERS: tl.constexpr = SCORE_BS // KV_BS
    else:
        BRANCH_ID: tl.constexpr = 3
        INNER_ITERS: tl.constexpr = (SCORE_BS_ORIG + KV_BS - 1) // KV_BS

    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H
    i_h_bi = tl.minimum(i_h, H_BI - 1)

    q_base = q_ptr + i_b * stride_q_b + i_t * stride_q_t + i_h * stride_q_h
    k_base = k_ptr + i_b * stride_k_b + i_h * stride_k_h
    v_base = v_ptr + i_b * stride_v_b + i_h * stride_v_h
    p_q = tl.make_block_ptr(
        base=q_base,
        shape=(D,),
        strides=(stride_q_d,),
        offsets=(0,),
        block_shape=(BD,),
        order=(0,),
    )
    b_q = tl.load(p_q, boundary_check=(0,)).to(tl.float32) * scale

    bi_base = bi_ptr + i_b * stride_bi_b + i_h_bi * stride_bi_h + i_t * stride_bi_n

    b_m = tl.full([], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([], dtype=tl.float32)
    b_o = tl.zeros([BV], dtype=tl.float32)

    for i_topk in range(topk):
        score_block_id = tl.load(bi_base + i_topk * stride_bi_k).to(tl.int32)

        if INNER_ITERS <= 16:
            for kv_sub_iter in tl.static_range(INNER_ITERS):
                b_m, b_acc, b_o = _flash_topk_attn_fwd_accumulate_sub(
                    kv_sub_iter, score_block_id,
                    k_base, v_base,
                    b_q,
                    stride_k_d, stride_k_t, stride_v_t, stride_v_d,
                    i_v,
                    BV, D, N, BD,
                    KV_BS, SCORE_BS_ORIG, SCORE_BS,
                    BRANCH_ID, NEG_INF,
                    b_m, b_acc, b_o,
                )
        else:
            for kv_sub_iter in range(INNER_ITERS):
                b_m, b_acc, b_o = _flash_topk_attn_fwd_accumulate_sub(
                    kv_sub_iter, score_block_id,
                    k_base, v_base,
                    b_q,
                    stride_k_d, stride_k_t, stride_v_t, stride_v_d,
                    i_v,
                    BV, D, N, BD,
                    KV_BS, SCORE_BS_ORIG, SCORE_BS,
                    BRANCH_ID, NEG_INF,
                    b_m, b_acc, b_o,
                )

    b_o = b_o / tl.where(b_acc > 0.0, b_acc, 1.0)

    p_o = tl.make_block_ptr(
        base=o_ptr + i_b * stride_o_b + i_t * stride_o_t + i_h * stride_o_h,
        shape=(D,),
        strides=(stride_o_d,),
        offsets=(i_v * BV,),
        block_shape=(BV,),
        order=(0,),
    )
    tl.store(p_o, b_o.to(o_ptr.dtype.element_ty), boundary_check=(0,))

    if i_v == 0:
        b_lse = b_m + tl.log(tl.where(b_acc > 0.0, b_acc, 1.0))
        lse_base = lse_ptr + i_b * stride_lse_b + i_h * stride_lse_h + i_t * stride_lse_n
        tl.store(lse_base, b_lse.to(lse_ptr.dtype.element_ty))


class _FlashTopKAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_indices, num_heads, block_size, scale):
        """Forward: Triton autotune over KV_BS × num_warps (single kernel entry)."""
        B, N, C = q.shape
        H = num_heads
        D = C // H
        topk = block_indices.shape[-1]

        SCORE_BS_ORIG = block_size
        SCORE_BS = _next_power_of_2(SCORE_BS_ORIG)
        IS_POW2 = SCORE_BS_ORIG == SCORE_BS
        BD = _next_power_of_2(D)
        BV = min(BD, 64)

        q_h = q.view(B, N, H, D)
        k_h = k.view(B, N, H, D)
        v_h = v.view(B, N, H, D)
        o_h = torch.empty_like(q_h)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

        H_BI = block_indices.shape[1]
        grid = (N, triton.cdiv(D, BV), B * H)
        _flash_topk_attn_fwd_kernel[grid](
            q_h, k_h, v_h, o_h, lse, block_indices,
            *q_h.stride(), *k_h.stride(), *v_h.stride(), *o_h.stride(),
            lse.stride(0), lse.stride(1), lse.stride(2),
            *block_indices.stride(),
            N=N, H=H, H_BI=H_BI,
            D=D, BD=BD, BV=BV,
            SCORE_BS_ORIG=SCORE_BS_ORIG, SCORE_BS=SCORE_BS,
            scale=scale, topk=topk,
        )
        KV_BS = _flash_topk_attn_fwd_kernel.best_config.kwargs["KV_BS"]
        if IS_POW2:
            if KV_BS >= SCORE_BS:
                BRANCH_ID, INNER_ITERS = 1, 1
            else:
                BRANCH_ID, INNER_ITERS = 2, SCORE_BS // KV_BS
        else:
            BRANCH_ID, INNER_ITERS = 3, (SCORE_BS_ORIG + KV_BS - 1) // KV_BS

        ctx.save_for_backward(q_h, k_h, v_h, o_h, block_indices, lse)
        ctx.meta = dict(
            H=H,
            D=D,
            BD=BD,
            BV=BV,
            SCORE_BS_ORIG=SCORE_BS_ORIG,
            SCORE_BS=SCORE_BS,
            KV_BS=KV_BS,
            INNER_ITERS=INNER_ITERS,
            BRANCH_ID=BRANCH_ID,
            IS_POW2=IS_POW2,
            scale=scale,
            topk=topk,
        )
        return o_h.view(B, N, C)

    @staticmethod
    def backward(ctx, do):
        _ = ctx.saved_tensors
        _ = ctx.meta
        _ = do
        raise NotImplementedError("flash_topk_attn backward is not implemented yet")


def flash_topk_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_indices: torch.Tensor,
    num_heads: int, block_size: int = 64,
    scale: Optional[float] = None, num_kv_heads: Optional[int] = None,
) -> torch.Tensor:
    """Sparse attention over top-k KV blocks (forward only in v0).

    KV tile size and num_warps are chosen via Triton autotune (see plan
    ``docs/plan/20260322_attention_optimizations.md``).

    Args:
        q: [B, N, C] query tensor, C = num_heads * D
        k: [B, N, C] key tensor
        v: [B, N, C] value tensor
        block_indices: [B, H_BI, N, topk] int32 with H_BI in {1, H}; if H_BI=1 the
            same indices are used for every head (no Python expand).
        num_heads: number of attention heads H
        block_size: score block size in tokens (any positive integer)
        scale: attention scale; defaults to 1/sqrt(D)
        num_kv_heads: reserved for GQA; must equal num_heads in v0

    Returns:
        o: [B, N, C] attention output
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "inputs must be on CUDA"
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    B, N, C = q.shape
    H = num_heads
    D = C // H
    assert C == H * D, f"C={C} must equal num_heads * D, got H={H}"
    assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"

    if num_kv_heads is not None and num_kv_heads != num_heads:
        raise NotImplementedError("GQA (num_kv_heads != num_heads) is not supported in v0")

    H_BI = block_indices.shape[1]
    assert H_BI == 1 or H_BI == H, (
        f"block_indices.shape[1] must be 1 or num_heads(H={H}), got {H_BI}"
    )

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_indices = block_indices.contiguous()

    return _FlashTopKAttnFunc.apply(q, k, v, block_indices, num_heads, block_size, scale)


def _flash_topk_attn_naive(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_indices: torch.Tensor,
    num_heads: int, block_size: int, scale: Optional[float] = None,
) -> torch.Tensor:
    """Sparse attention over top-k KV blocks (naive reference).

    Args:
        q: [B, N, C] query
        k: [B, N, C] key
        v: [B, N, C] value
        block_indices: [B, H, N, topk] int32 or int64, block ids in [0, M-1]
        num_heads: H
        block_size: tokens per block (SCORE_BS_ORIG)
        scale: default 1/sqrt(D)

    Returns:
        o: [B, N, C]
    """
    dtype = q.dtype
    B, N, C = q.shape
    H = num_heads
    D = C // H
    M = N // block_size
    topk = block_indices.shape[-1]

    assert k.shape == (B, N, C) and v.shape == (B, N, C)
    assert C == H * D
    assert N % block_size == 0
    H_BI = block_indices.shape[1]
    assert H_BI == 1 or H_BI == H
    assert block_indices.shape == (B, H_BI, N, topk)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = rearrange(q, "b n (h d) -> b h n d", h=H).float()
    k = rearrange(k, "b n (h d) -> b h n d", h=H).float()
    v = rearrange(v, "b n (h d) -> b h n d", h=H).float()
    bi = block_indices.long()
    if bi.shape[1] == 1 and H > 1:
        bi = bi.expand(B, H, N, topk)

    # [B, H, N, topk] -> for each (b,h,n), which block ids are selected.
    # -1 means empty slot and should not match any kv block.
    kv_block_id = torch.arange(N, device=q.device, dtype=torch.long) // block_size
    valid_slot = bi >= 0
    match = bi[..., None] == kv_block_id[None, None, None, :]
    valid = (match & valid_slot[..., None]).any(dim=-2)
    has_any = valid.any(dim=-1, keepdim=True)

    scores = torch.einsum("bhqd,bhkd->bhqk", q * scale, k)
    scores = scores.masked_fill(~valid, float("-inf"))
    safe_scores = torch.where(has_any, scores, torch.zeros_like(scores))
    attn = torch.softmax(safe_scores, dim=-1)
    attn = torch.where(has_any, attn, torch.zeros_like(attn))
    o = torch.einsum("bhqk,bhkd->bhqd", attn, v)

    o = rearrange(o, "b h n d -> b n (h d)")
    return o.to(dtype)
