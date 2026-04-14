"""Per-query per-score-block softmax probability mass."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

from flash_topk_attn_v2.block_score_cuda import flash_block_score_cuda

_CUDA_SUPPORTED_D = (32, 64, 96, 128, 160, 256)
_CUDA_MIN_D_KERNEL = 80


@triton.autotune(
    configs=[
        triton.Config({"Q_BS": q, "KV_TILE": kv}, num_warps=w)
        for q in [16, 32]
        for kv in [16, 32, 64, 128]
        for w in [4, 8]
    ],
    key=["N", "D", "SCORE_BS", "NUM_SCORE_BLOCKS"],
)
@triton.jit
def _block_local_score_kernel(
    Q,
    K,
    M_locals,
    L_locals,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    Q_BS: tl.constexpr,
    KV_TILE: tl.constexpr,
    SCORE_BS: tl.constexpr,
    PAD_HEAD: tl.constexpr,
    PAD_TAIL: tl.constexpr,
    NUM_SCORE_BLOCKS: tl.constexpr,
    stride_q_b,
    stride_q_h,
    stride_q_n,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_n,
    stride_k_d,
):
    i_t = tl.program_id(0)
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    q_start = i_t * Q_BS
    score_block_start = i_s * SCORE_BS - PAD_HEAD
    scale_log2 = (1.0 / tl.sqrt(tl.cast(D, tl.float32))) * 1.44269504
    NEG_INF: tl.constexpr = -1e38

    p_q = tl.make_block_ptr(
        base=Q + i_b * stride_q_b + i_h * stride_q_h,
        shape=(N, D),
        strides=(stride_q_n, stride_q_d),
        offsets=(q_start, 0),
        block_shape=(Q_BS, D_PAD),
        order=(1, 0),
    )
    if N % Q_BS == 0 and D == D_PAD:
        q = tl.load(p_q)
    elif N % Q_BS == 0:
        q = tl.load(p_q, boundary_check=(1,))
    elif D == D_PAD:
        q = tl.load(p_q, boundary_check=(0,))
    else:
        q = tl.load(p_q, boundary_check=(0, 1))

    m_local = tl.full((Q_BS,), float("-inf"), dtype=tl.float32)
    l_local = tl.zeros((Q_BS,), dtype=tl.float32)

    NUM_TILES: tl.constexpr = (SCORE_BS + KV_TILE - 1) // KV_TILE
    p_k = tl.make_block_ptr(
        base=K + i_b * stride_k_b + i_h * stride_k_h,
        shape=(D, N),
        strides=(stride_k_d, stride_k_n),
        offsets=(0, score_block_start),
        block_shape=(D_PAD, KV_TILE),
        order=(0, 1),
    )

    for tile_i in range(NUM_TILES):
        if D == D_PAD:
            k = tl.load(p_k, boundary_check=(1,))
        else:
            k = tl.load(p_k, boundary_check=(0, 1))

        s = tl.dot(q, k) * scale_log2

        if PAD_HEAD == 0 and PAD_TAIL == 0:
            if SCORE_BS % KV_TILE == 0:
                pass
            else:
                k_start = score_block_start + tile_i * KV_TILE
                kv_idx = k_start + tl.arange(0, KV_TILE)
                valid = kv_idx[None, :] < (score_block_start + SCORE_BS)
                s = tl.where(valid, s, NEG_INF)
        else:
            k_start = score_block_start + tile_i * KV_TILE
            kv_idx = k_start + tl.arange(0, KV_TILE)
            if SCORE_BS % KV_TILE == 0:
                valid = (kv_idx[None, :] >= 0) & (kv_idx[None, :] < N)
            else:
                in_seq = (kv_idx[None, :] >= 0) & (kv_idx[None, :] < N)
                in_blk = kv_idx[None, :] < (score_block_start + SCORE_BS)
                valid = in_seq & in_blk
            s = tl.where(valid, s, NEG_INF)

        if PAD_HEAD == 0 and PAD_TAIL == 0:
            if SCORE_BS % KV_TILE == 0:
                m_tile = tl.max(s, axis=1)
                m_new = tl.maximum(m_local, m_tile)
                alpha = tl.exp2(m_local - m_new)
                p_sum = tl.sum(tl.exp2(s - m_new[:, None]), axis=1)
            else:
                valid_any = tl.max(valid.to(tl.int32), axis=1) > 0
                m_tile = tl.where(valid_any, tl.max(s, axis=1), float("-inf"))
                m_new = tl.maximum(m_local, m_tile)
                alpha = tl.exp2(m_local - m_new)
                p_sum = tl.sum(tl.exp2(tl.where(valid, s - m_new[:, None], NEG_INF)), axis=1)
        else:
            valid_any = tl.max(valid.to(tl.int32), axis=1) > 0
            m_tile = tl.where(valid_any, tl.max(s, axis=1), float("-inf"))
            m_new = tl.maximum(m_local, m_tile)
            alpha = tl.exp2(m_local - m_new)
            p_sum = tl.sum(tl.exp2(tl.where(valid, s - m_new[:, None], NEG_INF)), axis=1)
        l_local = l_local * alpha + p_sum
        m_local = m_new

        p_k = tl.advance(p_k, (0, KV_TILE))

    p_m_out = tl.make_block_ptr(
        base=M_locals + i_b * H * N * NUM_SCORE_BLOCKS + i_h * N * NUM_SCORE_BLOCKS + i_s,
        shape=(N,),
        strides=(NUM_SCORE_BLOCKS,),
        offsets=(q_start,),
        block_shape=(Q_BS,),
        order=(0,),
    )
    p_l_out = tl.make_block_ptr(
        base=L_locals + i_b * H * N * NUM_SCORE_BLOCKS + i_h * N * NUM_SCORE_BLOCKS + i_s,
        shape=(N,),
        strides=(NUM_SCORE_BLOCKS,),
        offsets=(q_start,),
        block_shape=(Q_BS,),
        order=(0,),
    )
    if N % Q_BS == 0:
        tl.store(p_m_out, m_local)
        tl.store(p_l_out, l_local)
    else:
        tl.store(p_m_out, m_local, boundary_check=(0,))
        tl.store(p_l_out, l_local, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({"Q_BS_B": q, "TILE_M": tm}, num_warps=w)
        for q in [16, 32, 64]
        for tm in [16, 32, 64, 128]
        for w in [4, 8]
    ],
    key=["N", "NUM_SCORE_BLOCKS"],
)
@triton.jit
def _block_normalize_kernel(
    M_locals,
    L_locals,
    BlockScores,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    Q_BS_B: tl.constexpr,
    TILE_M: tl.constexpr,
    NUM_SCORE_BLOCKS: tl.constexpr,
    stride_m_b,
    stride_m_h,
    stride_m_n,
    stride_m_m,
    stride_out_b,
    stride_out_h,
    stride_out_n,
    stride_out_m,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = tl.program_id(2)

    q_start = i_t * Q_BS_B

    p_m = tl.make_block_ptr(
        base=M_locals + i_b * stride_m_b + i_h * stride_m_h,
        shape=(N, NUM_SCORE_BLOCKS),
        strides=(stride_m_n, stride_m_m),
        offsets=(q_start, 0),
        block_shape=(Q_BS_B, TILE_M),
        order=(1, 0),
    )
    p_l = tl.make_block_ptr(
        base=L_locals + i_b * stride_m_b + i_h * stride_m_h,
        shape=(N, NUM_SCORE_BLOCKS),
        strides=(stride_m_n, stride_m_m),
        offsets=(q_start, 0),
        block_shape=(Q_BS_B, TILE_M),
        order=(1, 0),
    )

    m_global = tl.full((Q_BS_B,), float("-inf"), dtype=tl.float32)
    L_global = tl.zeros((Q_BS_B,), dtype=tl.float32)

    NUM_TILES_M: tl.constexpr = (NUM_SCORE_BLOCKS + TILE_M - 1) // TILE_M
    for _ in range(NUM_TILES_M):
        if N % Q_BS_B == 0 and NUM_SCORE_BLOCKS % TILE_M == 0:
            m_tile = tl.load(p_m)
            l_tile = tl.load(p_l)
        elif N % Q_BS_B == 0:
            m_tile = tl.load(p_m, boundary_check=(1,))
            l_tile = tl.load(p_l, boundary_check=(1,))
        elif NUM_SCORE_BLOCKS % TILE_M == 0:
            m_tile = tl.load(p_m, boundary_check=(0,))
            l_tile = tl.load(p_l, boundary_check=(0,))
        else:
            m_tile = tl.load(p_m, boundary_check=(0, 1))
            l_tile = tl.load(p_l, boundary_check=(0, 1))

        m_batch = tl.max(m_tile, axis=1)
        beta_tile = tl.exp2(m_tile - m_batch[:, None])
        l_batch = tl.sum(l_tile * beta_tile, axis=1)

        m_new = tl.maximum(m_global, m_batch)
        alpha = tl.exp2(m_global - m_new)
        gamma = tl.exp2(m_batch - m_new)
        L_global = L_global * alpha + l_batch * gamma
        m_global = m_new

        p_m = tl.advance(p_m, (0, TILE_M))
        p_l = tl.advance(p_l, (0, TILE_M))

    safe_L = tl.where(L_global > 0.0, L_global, 1.0)

    p_m = tl.advance(p_m, (0, -NUM_TILES_M * TILE_M))
    p_l = tl.advance(p_l, (0, -NUM_TILES_M * TILE_M))
    p_out = tl.make_block_ptr(
        base=BlockScores + i_b * stride_out_b + i_h * stride_out_h,
        shape=(N, NUM_SCORE_BLOCKS),
        strides=(stride_out_n, stride_out_m),
        offsets=(q_start, 0),
        block_shape=(Q_BS_B, TILE_M),
        order=(1, 0),
    )

    for _ in range(NUM_TILES_M):
        if N % Q_BS_B == 0 and NUM_SCORE_BLOCKS % TILE_M == 0:
            m_tile = tl.load(p_m)
            l_tile = tl.load(p_l)
        elif N % Q_BS_B == 0:
            m_tile = tl.load(p_m, boundary_check=(1,))
            l_tile = tl.load(p_l, boundary_check=(1,))
        elif NUM_SCORE_BLOCKS % TILE_M == 0:
            m_tile = tl.load(p_m, boundary_check=(0,))
            l_tile = tl.load(p_l, boundary_check=(0,))
        else:
            m_tile = tl.load(p_m, boundary_check=(0, 1))
            l_tile = tl.load(p_l, boundary_check=(0, 1))

        beta = tl.exp2(m_tile - m_global[:, None])
        block_score = l_tile * beta / safe_L[:, None]

        if N % Q_BS_B == 0 and NUM_SCORE_BLOCKS % TILE_M == 0:
            tl.store(p_out, block_score)
        elif N % Q_BS_B == 0:
            tl.store(p_out, block_score, boundary_check=(1,))
        elif NUM_SCORE_BLOCKS % TILE_M == 0:
            tl.store(p_out, block_score, boundary_check=(0,))
        else:
            tl.store(p_out, block_score, boundary_check=(0, 1))

        p_m = tl.advance(p_m, (0, TILE_M))
        p_l = tl.advance(p_l, (0, TILE_M))
        p_out = tl.advance(p_out, (0, TILE_M))


def _flash_block_score_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int | None,
    score_block_size: int,
    padding: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Reference softmax over keys aggregated into virtual score blocks.

    Args:
        q: Query tensor ``[B, N, C]`` or ``[B, H, N, D]``.
        k: Key tensor, same layout as ``q`` after any 4D flattening.
        num_heads: Head count; required for 3D input, optional for 4D (must match ``H``).
        score_block_size: Tokens per score block.
        padding: ``(pad_head, pad_tail)``; ``pad_head + N + pad_tail`` must divide
            ``score_block_size``.

    Returns:
        Per-query block masses ``[B, H, N, M]`` float32, summing to 1 over ``M``.
    """
    input_4d = q.ndim == 4
    if input_4d:
        B, H, N, D = q.shape
        if num_heads is not None and num_heads != H:
            raise ValueError(f"num_heads ({num_heads}) != q.shape[1] ({H})")
        num_heads = H
        C = H * D
        q = q.transpose(1, 2).reshape(B, N, C)
        k = k.transpose(1, 2).reshape(B, N, C)
    else:
        B, N, C = q.shape
        if num_heads is None:
            raise ValueError("num_heads is required for 3D [B, N, C] input")

    H = num_heads
    D = C // H
    pad_head, pad_tail = padding
    if pad_head < 0 or pad_tail < 0:
        raise ValueError("padding must be non-negative (pad_head, pad_tail)")
    if pad_head >= score_block_size or pad_tail >= score_block_size:
        raise ValueError(
            f"padding ({pad_head}, {pad_tail}) must each be < score_block_size "
            f"({score_block_size})"
        )
    n_padded = pad_head + N + pad_tail
    if n_padded % score_block_size != 0:
        raise ValueError(
            f"pad_head + N + pad_tail = {n_padded} must be divisible by "
            f"score_block_size ({score_block_size})"
        )
    M = n_padded // score_block_size

    if k.shape != (B, N, C):
        raise ValueError(f"k shape {k.shape} != {(B, N, C)}")
    if C % H != 0:
        raise ValueError(f"C={C} not divisible by num_heads={H}")

    scale = D ** -0.5
    qf = rearrange(q, "b n (h d) -> b h n d", h=H).float()
    kf = rearrange(k, "b n (h d) -> b h n d", h=H).float()
    scores = torch.einsum("bhqd,bhkd->bhqk", qf * scale, kf)
    attn = torch.softmax(scores, dim=-1)

    out = torch.zeros(B, H, N, M, device=q.device, dtype=torch.float32)
    for m in range(M):
        start = m * score_block_size - pad_head
        end = (m + 1) * score_block_size - pad_head
        start = max(0, start)
        end = min(N, end)
        if start < end:
            out[..., m] = attn[..., start:end].sum(dim=-1)
    return out


def _parse_block_score_args(q, k, num_heads, score_block_size, padding):
    if q.device.type != "cuda":
        raise ValueError("flash_block_score requires CUDA tensors")

    input_4d = q.ndim == 4
    if input_4d:
        B, H, N, D = q.shape
        if num_heads != H:
            raise ValueError(f"num_heads ({num_heads}) != q.shape[1] ({H})")
        C = H * D
    else:
        if q.ndim != 3:
            raise ValueError("q must be 3D [B, N, C] or 4D [B, H, N, D]")
        B, N, C = q.shape

    H = num_heads
    D = C // H
    pad_head, pad_tail = padding
    if pad_head < 0 or pad_tail < 0:
        raise ValueError("padding must be non-negative (pad_head, pad_tail)")
    if pad_head >= score_block_size or pad_tail >= score_block_size:
        raise ValueError(
            f"padding ({pad_head}, {pad_tail}) must each be < score_block_size "
            f"({score_block_size})"
        )
    n_padded = pad_head + N + pad_tail
    if n_padded % score_block_size != 0:
        need = score_block_size - (n_padded % score_block_size)
        raise ValueError(
            f"pad_head + N + pad_tail = {n_padded} must be divisible by "
            f"score_block_size ({score_block_size}). Try padding=(0, {pad_tail + need})."
        )

    if input_4d:
        if k.shape != (B, H, N, D):
            raise ValueError(f"k shape {k.shape} != {(B, H, N, D)}")
    else:
        if k.shape != (B, N, C):
            raise ValueError(f"k shape {k.shape} != {(B, N, C)}")
    if C % H != 0:
        raise ValueError(f"C={C} not divisible by num_heads={H}")

    if q.dtype != k.dtype:
        raise ValueError(f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})")

    return input_4d, B, H, N, D, C, n_padded, pad_head, pad_tail


def flash_block_score(
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    score_block_size: int,
    padding: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Fused block-wise attention mass per query and score block.

    Automatically dispatches to CUDA (CuTe MMA) when D_kernel >= 80,
    otherwise falls back to Triton.

    Args:
        q: ``[B, N, C]`` or ``[B, H, N, D]`` on CUDA (float16, bfloat16, or float32).
        k: Same shape, layout, and dtype as ``q``.
        num_heads: Must equal ``H`` when ``q`` is 4D.
        score_block_size: Logical KV tokens per score block (any positive integer).
        padding: Virtual padding ``(pad_head, pad_tail)``, each strictly less than
            ``score_block_size``. Padded length must divide ``score_block_size``.

    Returns:
        ``[B, H, N, M]`` float32 with ``M = (pad_head + N + pad_tail) // score_block_size``;
        each row over ``M`` sums to 1.
    """
    parsed = _parse_block_score_args(q, k, num_heads, score_block_size, padding)
    input_4d, B, H, N, D, C, n_padded, pad_head, pad_tail = parsed

    D_kernel = next((sd for sd in _CUDA_SUPPORTED_D if sd >= D), None)
    if D_kernel is not None and D_kernel >= _CUDA_MIN_D_KERNEL:
        return flash_block_score_cuda(q, k, num_heads, score_block_size, padding)

    return _run_triton_block_score(
        q, k, input_4d, B, H, N, D, n_padded, pad_head, pad_tail,
        score_block_size,
    )


def _flash_block_score_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    score_block_size: int,
    padding: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Pure Triton backend, same signature as ``flash_block_score``."""
    parsed = _parse_block_score_args(q, k, num_heads, score_block_size, padding)
    input_4d, B, H, N, D, C, n_padded, pad_head, pad_tail = parsed
    return _run_triton_block_score(
        q, k, input_4d, B, H, N, D, n_padded, pad_head, pad_tail,
        score_block_size,
    )


def _run_triton_block_score(
    q, k, input_4d, B, H, N, D, n_padded, pad_head, pad_tail,
    score_block_size,
):
    SCORE_BS = score_block_size
    num_score_blocks = n_padded // SCORE_BS
    d_pad = triton.next_power_of_2(D)

    if input_4d:
        q_4d = q.contiguous()
        k_4d = k.contiguous()
    else:
        q_4d = q.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        k_4d = k.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()

    M_locals = torch.empty(
        B, H, N, num_score_blocks, device=q.device, dtype=torch.float32
    )
    L_locals = torch.empty(
        B, H, N, num_score_blocks, device=q.device, dtype=torch.float32
    )
    block_scores = torch.empty(
        B, H, N, num_score_blocks, device=q.device, dtype=torch.float32
    )

    grid_a = lambda meta: (triton.cdiv(N, meta["Q_BS"]), num_score_blocks, B * H)
    _block_local_score_kernel[grid_a](
        q_4d,
        k_4d,
        M_locals,
        L_locals,
        B=B,
        N=N,
        H=H,
        D=D,
        D_PAD=d_pad,
        SCORE_BS=SCORE_BS,
        PAD_HEAD=pad_head,
        PAD_TAIL=pad_tail,
        NUM_SCORE_BLOCKS=num_score_blocks,
        stride_q_b=q_4d.stride(0),
        stride_q_h=q_4d.stride(1),
        stride_q_n=q_4d.stride(2),
        stride_q_d=q_4d.stride(3),
        stride_k_b=k_4d.stride(0),
        stride_k_h=k_4d.stride(1),
        stride_k_n=k_4d.stride(2),
        stride_k_d=k_4d.stride(3),
    )

    grid_b = lambda meta: (triton.cdiv(N, meta["Q_BS_B"]), H, B)
    _block_normalize_kernel[grid_b](
        M_locals,
        L_locals,
        block_scores,
        B=B,
        N=N,
        H=H,
        NUM_SCORE_BLOCKS=num_score_blocks,
        stride_m_b=M_locals.stride(0),
        stride_m_h=M_locals.stride(1),
        stride_m_n=M_locals.stride(2),
        stride_m_m=M_locals.stride(3),
        stride_out_b=block_scores.stride(0),
        stride_out_h=block_scores.stride(1),
        stride_out_n=block_scores.stride(2),
        stride_out_m=block_scores.stride(3),
    )

    return block_scores


__all__ = [
    "flash_block_score",
    "_flash_block_score_naive",
    "_flash_block_score_triton",
]
