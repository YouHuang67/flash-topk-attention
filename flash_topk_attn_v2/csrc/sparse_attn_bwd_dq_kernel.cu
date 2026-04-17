#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "sparse_attn_bwd_dq_kernel.cuh"


template <int kBlockM, int kBlockN, int kHeadDim, int kNWarps, typename Element>
__global__ void __launch_bounds__(kNWarps * 32, 1)
sparse_attn_bwd_dq_kernel(
    const Element* __restrict__ q_ptr,
    const Element* __restrict__ k_ptr,
    const Element* __restrict__ v_ptr,
    const Element* __restrict__ do_ptr,
    const int32_t* __restrict__ merged_indices_ptr,
    const int32_t* __restrict__ counts_ptr,
    const float* __restrict__ softmax_max_ptr,
    const float* __restrict__ softmax_lse_ptr,
    const float* __restrict__ delta_ptr,
    Element* __restrict__ dq_ptr,
    int B, int N, int N_phys, int H, int D,
    int QM, int qblock_topk,
    int q_block_size, int kv_block_size,
    int stride_q_b, int stride_q_n, int stride_q_h,
    int stride_k_b, int stride_k_n, int stride_k_h,
    int stride_v_b, int stride_v_n, int stride_v_h,
    int stride_do_b, int stride_do_n, int stride_do_h,
    int stride_dq_b, int stride_dq_n, int stride_dq_h,
    float softmax_scale_log2,
    float softmax_scale,
    int q_pad_head, int kv_pad_head
) {
    using Types = SparseAttnBwdDqTypes<kBlockM, kBlockN, kHeadDim, kNWarps, Element>;
    using SharedStorage = typename Types::SharedStorage;

    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const qm_idx = blockIdx.x;
    int const bh_idx = blockIdx.z;
    int const b_idx = bh_idx / H;
    int const h_idx = bh_idx % H;
    int const thread_idx = threadIdx.x;

    int const count = __ldg(counts_ptr + b_idx * H * QM + h_idx * QM + qm_idx);
    if (count <= 0) {
        int const q_global_start = qm_idx * q_block_size - q_pad_head;
        for (int row = thread_idx; row < q_block_size; row += kNWarps * 32) {
            int const global_row = q_global_start + row;
            if (global_row >= 0 && global_row < N) {
                for (int d = 0; d < kHeadDim; d++) {
                    dq_ptr[b_idx * stride_dq_b + global_row * stride_dq_n
                           + h_idx * stride_dq_h + d] = Element(0);
                }
            }
        }
        return;
    }

    int64_t indices_base = (int64_t)b_idx * H * QM * qblock_topk
                         + (int64_t)h_idx * QM * qblock_topk
                         + (int64_t)qm_idx * qblock_topk;
    const int32_t* block_indices_ptr = merged_indices_ptr + indices_base;

    using TiledMma = typename Types::TiledMma;
    using SmemLayoutQ = typename Types::SmemLayoutQ;
    using SmemLayoutdO = typename Types::SmemLayoutdO;
    using SmemLayoutK = typename Types::SmemLayoutK;
    using SmemLayoutV = typename Types::SmemLayoutV;
    using SmemLayoutKt = typename Types::SmemLayoutKt;
    using SmemLayoutdQ = typename Types::SmemLayoutdQ;
    using SmemCopyAtom = typename Types::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Types::SmemCopyAtomTransposed;
    using GmemTiledCopyQKV = typename Types::GmemTiledCopyQKV;
    using SmemCopyAtomO = typename Types::SmemCopyAtomO;
    static constexpr int kNRows = Types::kNRows;

    Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(smem.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(smem.smem_v.data()), SmemLayoutV{});
    Tensor sKt = make_tensor(make_smem_ptr(smem.smem_k.data()), SmemLayoutKt{});

    int const q_start = qm_idx * q_block_size - q_pad_head;
    int q_load_start = q_start;
    if (q_load_start < 0) q_load_start = 0;
    if (q_load_start + kBlockM > N_phys) q_load_start = N_phys - kBlockM;
    int const q_row_offset = q_start - q_load_start;

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);

    {
        int const q_offset = b_idx * stride_q_b + q_load_start * stride_q_n
                           + h_idx * stride_q_h;
        Tensor mQ = make_tensor(
            make_gmem_ptr(q_ptr + q_offset),
            Shape<Int<kBlockM>, Int<kHeadDim>>{},
            make_stride(stride_q_n, _1{}));
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(mQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
        cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    }
    cute::cp_async_fence();

    {
        int const do_offset = b_idx * stride_do_b + q_load_start * stride_do_n
                            + h_idx * stride_do_h;
        Tensor mdO = make_tensor(
            make_gmem_ptr(do_ptr + do_offset),
            Shape<Int<kBlockM>, Int<kHeadDim>>{},
            make_stride(stride_do_n, _1{}));
        Tensor tDOgDO = gmem_thr_copy_QKV.partition_S(mdO);
        Tensor tDOsDO = gmem_thr_copy_QKV.partition_D(sdO);
        cute::copy(gmem_tiled_copy_QKV, tDOgDO, tDOsDO);
    }
    cute::cp_async_fence();

    using TensorF = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorF reg_softmax_max;
    TensorF reg_softmax_lse;
    TensorF reg_delta;

    sparse_cp_async_wait<0>();
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrdO = thr_mma.partition_fragment_A(sdO);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    auto smem_tiled_copy_Bt = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_Bt = smem_tiled_copy_Bt.get_thread_slice(thread_idx);

    Tensor tSsQ = smem_thr_copy_A.partition_S(sQ);
    Tensor tSsdO = smem_thr_copy_A.partition_S(sdO);
    Tensor tSsK = smem_thr_copy_B.partition_S(sK);
    Tensor tSsV = smem_thr_copy_B.partition_S(sV);
    Tensor tOsKt = smem_thr_copy_Bt.partition_S(sKt);

    {
        Tensor tSrQ_copy_view = smem_thr_copy_A.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_A, tSsQ, tSrQ_copy_view);
    }
    {
        Tensor tSrdO_copy_view = smem_thr_copy_A.retile_D(tSrdO);
        cute::copy(smem_tiled_copy_A, tSsdO, tSrdO_copy_view);
    }

    auto taccScO = thr_mma.partition_C(
        cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}));
    auto taccScO_rowcol = make_tensor(taccScO.data(),
        SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(taccScO.layout()));

    {
        auto taccOcO_hd = thr_mma.partition_C(
            cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
        auto taccOcO_hd_rowcol = make_tensor(taccOcO_hd.data(),
            SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(taccOcO_hd.layout()));
        auto taccOcO_hd_row = taccOcO_hd_rowcol(_, _0{});

        #pragma unroll
        for (int mi = 0; mi < size(taccOcO_hd_row); ++mi) {
            int const tile_row = get<0>(taccOcO_hd_row(mi));
            int const global_row = q_start + (tile_row - q_row_offset);
            bool valid_row = (tile_row >= q_row_offset
                              && tile_row < q_row_offset + q_block_size
                              && global_row >= 0 && global_row < N);
            if (valid_row) {
                int const lse_offset = b_idx * H * N + h_idx * N + global_row;
                reg_softmax_max(mi) = softmax_max_ptr[lse_offset];
                reg_softmax_lse(mi) = softmax_lse_ptr[lse_offset];
                reg_delta(mi) = delta_ptr[lse_offset];
            } else {
                reg_softmax_max(mi) = -INFINITY;
                reg_softmax_lse(mi) = -INFINITY;
                reg_delta(mi) = 0.f;
            }
        }
    }

    Tensor tOrDQ = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(tOrDQ);

    int const kv_tiles = (kv_block_size + kBlockN - 1) / kBlockN;
    int const total_iters = count * kv_tiles;

    auto load_K = [&](int idx, int sub) {
        int n_block = block_indices_ptr[idx];
        int token_start = n_block * kv_block_size + sub * kBlockN - kv_pad_head;
        if (token_start < 0) token_start = 0;
        if (token_start + kBlockN > N_phys) token_start = N_phys - kBlockN;
        int k_offset = b_idx * stride_k_b + token_start * stride_k_n
                     + h_idx * stride_k_h;
        Tensor gK = make_tensor(
            make_gmem_ptr(k_ptr + k_offset),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            make_stride(stride_k_n, _1{}));
        Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
        cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
        cute::cp_async_fence();
    };

    auto load_V = [&](int idx, int sub) {
        int n_block = block_indices_ptr[idx];
        int token_start = n_block * kv_block_size + sub * kBlockN - kv_pad_head;
        if (token_start < 0) token_start = 0;
        if (token_start + kBlockN > N_phys) token_start = N_phys - kBlockN;
        int v_offset = b_idx * stride_v_b + token_start * stride_v_n
                     + h_idx * stride_v_h;
        Tensor gV = make_tensor(
            make_gmem_ptr(v_ptr + v_offset),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            make_stride(stride_v_n, _1{}));
        Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
        Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
        cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        cute::cp_async_fence();
    };

    load_K(0, 0);
    load_V(0, 0);

    for (int iter = 0; iter < total_iters; ++iter) {
        int const idx = iter / kv_tiles;
        int const sub = iter % kv_tiles;

        int const n_block = block_indices_ptr[idx];
        int const intended_start = n_block * kv_block_size
                                 + sub * kBlockN - kv_pad_head;
        int actual_start = intended_start;
        if (actual_start < 0) actual_start = 0;
        if (actual_start + kBlockN > N_phys) actual_start = N_phys - kBlockN;
        int const col_offset = intended_start - actual_start;
        int const valid_cols = min(kBlockN, kv_block_size - sub * kBlockN);

        sparse_cp_async_wait<0>();
        __syncthreads();

        Tensor tSrS = partition_fragment_C(
            tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrS);

        {
            Tensor tSrQ_cur = tSrQ;
            Tensor tSrK = thr_mma.partition_fragment_B(sK);

            sparse_gemm_sm80<true>(
                tSrS, tSrQ_cur, tSrK, tSsQ, tSsK,
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                smem_thr_copy_A, smem_thr_copy_B, nullptr);
        }

        bool need_col_mask = (col_offset > 0 || valid_cols < kBlockN
                              || intended_start < 0
                              || intended_start + valid_cols > N);
        if (need_col_mask) {
            Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrS.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                    int col = get<1>(taccScO_rowcol(mi, ni));
                    int kv_global = intended_start + (col - col_offset);
                    if (col < col_offset || col >= col_offset + valid_cols
                        || kv_global < 0 || kv_global >= N) {
                        tSrS_rowcol(mi, ni) = -INFINITY;
                    }
                }
            }
        }

        if (q_block_size < kBlockM || q_pad_head > 0) {
            Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrS.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                int row = get<0>(taccScO_rowcol(mi, _0{}));
                int global_row = q_start + (row - q_row_offset);
                if (row < q_row_offset
                    || row >= q_row_offset + q_block_size
                    || global_row < 0
                    || global_row >= N) {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                        tSrS_rowcol(mi, ni) = -INFINITY;
                    }
                }
            }
        }

        {
            Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrS.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                float max_val = reg_softmax_max(mi);
                float lse_val = reg_softmax_lse(mi);
                #pragma unroll
                for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                    float s_scaled = tSrS_rowcol(mi, ni) * softmax_scale;
                    tSrS_rowcol(mi, ni) = (s_scaled == -INFINITY)
                        ? 0.f : expf(s_scaled - max_val - lse_val);
                }
            }
        }

        Tensor tSrdP = partition_fragment_C(
            tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrdP);

        {
            Tensor tSrdO_cur = tSrdO;
            Tensor tSrV = thr_mma.partition_fragment_B(sV);
            sparse_gemm_sm80<true>(
                tSrdP, tSrdO_cur, tSrV, tSsdO, tSsV,
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                smem_thr_copy_A, smem_thr_copy_B, nullptr);
        }

        {
            Tensor tP_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrS.layout()));
            Tensor tdP_rowcol = make_tensor(tSrdP.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrdP.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tP_rowcol); ++mi) {
                float delta_i = reg_delta(mi);
                #pragma unroll
                for (int ni = 0; ni < size<1>(tP_rowcol); ++ni) {
                    float p_val = tP_rowcol(mi, ni);
                    tP_rowcol(mi, ni) = p_val * (tdP_rowcol(mi, ni) - delta_i);
                }
            }
        }

        using LayoutAcc = decltype(tSrS.layout());
        auto convert_layout_acc_Aregs_sm80 = [](LayoutAcc acc_layout) {
            auto l = logical_divide(
                acc_layout, Shape<Underscore, Underscore, _2>{});
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)),
                               get<1>(l), get<2, 1>(l));
        };

        Tensor tDSregs = make_tensor(tSrS.data(),
            convert_layout_acc_Aregs_sm80(tSrS.layout()));
        Tensor tDSelem = make_tensor_like<Element>(tDSregs);
        sparse_convert_type_out<Element>(tDSregs, tDSelem);

        Tensor tOrKt = thr_mma.partition_fragment_B(sKt);

        sparse_gemm_rs_sm80(tOrDQ, tDSelem, tOrKt, tOsKt,
                            tiled_mma, smem_tiled_copy_Bt, smem_thr_copy_Bt);

        __syncthreads();

        if (iter + 1 < total_iters) {
            int const next_idx = (iter + 1) / kv_tiles;
            int const next_sub = (iter + 1) % kv_tiles;
            load_K(next_idx, next_sub);
            load_V(next_idx, next_sub);
        }
    }

    {
        Tensor dQ_rowcol = make_tensor(tOrDQ.data(),
            SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                tOrDQ.layout()));
        #pragma unroll
        for (int mi = 0; mi < size<0>(dQ_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dQ_rowcol); ++ni) {
                dQ_rowcol(mi, ni) *= softmax_scale;
            }
        }
    }

    __syncthreads();

    Tensor sdQ = make_tensor(
        make_smem_ptr(smem.smem_dq.data()), SmemLayoutdQ{});

    {
        auto smem_tiled_copy_dQ = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(thread_idx);

        Tensor tOrDQ_out = make_tensor_like<Element>(tOrDQ);
        sparse_convert_type_out<Element>(tOrDQ, tOrDQ_out);

        Tensor taccOrdQ = smem_thr_copy_dQ.retile_S(tOrDQ_out);
        Tensor taccOsdQ = smem_thr_copy_dQ.partition_D(sdQ);
        cute::copy(smem_tiled_copy_dQ, taccOrdQ, taccOsdQ);
    }
    __syncthreads();

    {
        for (int row = thread_idx; row < q_block_size; row += kNWarps * 32) {
            int const global_row = q_start + row;
            if (global_row >= 0 && global_row < N) {
                int const dq_base = b_idx * stride_dq_b
                                  + global_row * stride_dq_n
                                  + h_idx * stride_dq_h;
                for (int d = 0; d < kHeadDim; d++) {
                    dq_ptr[dq_base + d] = sdQ(q_row_offset + row, d);
                }
            }
        }
    }
}


void sparse_attn_bwd_dq_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor do_tensor,
    torch::Tensor merged_indices, torch::Tensor counts,
    torch::Tensor softmax_max, torch::Tensor softmax_lse,
    torch::Tensor delta,
    torch::Tensor dq,
    int B, int N, int N_phys, int H, int D,
    int QM, int qblock_topk,
    int q_block_size, int kv_block_size,
    float softmax_scale,
    int q_pad_head, int kv_pad_head
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    int kNWarps;
    if (q_block_size <= 16)       kNWarps = 1;
    else if (q_block_size <= 32)  kNWarps = 2;
    else                          kNWarps = 4;

    dim3 grid(QM, 1, B * H);
    dim3 block(kNWarps * 32);

    float softmax_scale_log2 = softmax_scale * float(M_LOG2E);

    int stride_q_b = q.stride(0);
    int stride_q_n = q.stride(1);
    int stride_k_b = k.stride(0);
    int stride_k_n = k.stride(1);
    int stride_v_b = v.stride(0);
    int stride_v_n = v.stride(1);
    int stride_do_b = do_tensor.stride(0);
    int stride_do_n = do_tensor.stride(1);
    int stride_dq_b = dq.stride(0);
    int stride_dq_n = dq.stride(1);

    bool is_bf16 = (q.scalar_type() == at::ScalarType::BFloat16);

    #define LAUNCH_DQ(BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS, Elem)                 \
        do {                                                                     \
            using SharedStorage = typename SparseAttnBwdDqTypes<                 \
                BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS, Elem>::SharedStorage;       \
            int smem_size = sizeof(SharedStorage);                               \
            auto kernel = sparse_attn_bwd_dq_kernel<                            \
                BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS, Elem>;                     \
            if (smem_size > 48 * 1024) {                                         \
                cudaFuncSetAttribute(                                             \
                    kernel,                                                       \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);     \
            }                                                                     \
            kernel<<<grid, block, smem_size, stream>>>(                          \
                reinterpret_cast<const Elem*>(q.data_ptr()),                     \
                reinterpret_cast<const Elem*>(k.data_ptr()),                     \
                reinterpret_cast<const Elem*>(v.data_ptr()),                     \
                reinterpret_cast<const Elem*>(do_tensor.data_ptr()),             \
                merged_indices.data_ptr<int32_t>(),                              \
                counts.data_ptr<int32_t>(),                                      \
                softmax_max.data_ptr<float>(),                                   \
                softmax_lse.data_ptr<float>(),                                   \
                delta.data_ptr<float>(),                                         \
                reinterpret_cast<Elem*>(dq.data_ptr()),                          \
                B, N, N_phys, H, D, QM, qblock_topk,                            \
                q_block_size, kv_block_size,                                     \
                stride_q_b, stride_q_n, D,                                       \
                stride_k_b, stride_k_n, D,                                       \
                stride_v_b, stride_v_n, D,                                       \
                stride_do_b, stride_do_n, D,                                     \
                stride_dq_b, stride_dq_n, D,                                     \
                softmax_scale_log2, softmax_scale,                               \
                q_pad_head, kv_pad_head);                                        \
        } while (0)

    #define DISPATCH_DQ_DTYPE(BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS)               \
        do {                                                                     \
            if (is_bf16) { LAUNCH_DQ(BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS,       \
                                     cutlass::bfloat16_t); }                    \
            else         { LAUNCH_DQ(BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS,       \
                                     cutlass::half_t); }                        \
        } while (0)

    #define DISPATCH_DQ_NWARPS(HEAD_DIM)                                        \
        do {                                                                     \
            if (kNWarps == 1)      DISPATCH_DQ_DTYPE(16,  64, HEAD_DIM, 1);    \
            else if (kNWarps == 2) DISPATCH_DQ_DTYPE(32,  64, HEAD_DIM, 2);    \
            else                   DISPATCH_DQ_DTYPE(64,  64, HEAD_DIM, 4);    \
        } while (0)

    if (D == 128)          DISPATCH_DQ_NWARPS(128);
    else if (D == 64)      DISPATCH_DQ_NWARPS(64);
    else if (D == 256)     DISPATCH_DQ_NWARPS(256);
    else if (D == 32)      DISPATCH_DQ_NWARPS(32);
    else if (D == 96)      DISPATCH_DQ_NWARPS(96);
    else if (D == 160)     DISPATCH_DQ_NWARPS(160);

    #undef LAUNCH_DQ
    #undef DISPATCH_DQ_DTYPE
    #undef DISPATCH_DQ_NWARPS
}
