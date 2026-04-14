#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "sparse_attn_kernel.cuh"


template <int kBlockM, int kBlockN, int kHeadDim, int kNWarps, typename Element>
struct BlockScoreTypes {
    using Types = SparseAttnTypes<kBlockM, kBlockN, kHeadDim, kNWarps, Element>;
    using SmemLayoutQ = typename Types::SmemLayoutQ;
    using SmemLayoutK = typename Types::SmemLayoutK;

    struct SharedStorage {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };
};


template <int kBlockM, int kBlockN, int kHeadDim, int kNWarps, typename Element>
__global__ void __launch_bounds__(kNWarps * 32, 2)
block_score_kernel(
    const Element* __restrict__ q_ptr,
    const Element* __restrict__ k_ptr,
    float* __restrict__ m_locals,
    float* __restrict__ l_locals,
    float* __restrict__ block_scores,
    int N, int N_phys, int M,
    int score_block_size,
    int sb_per_group,
    int num_groups,
    int stride_qk_bh, int stride_qk_n,
    float softmax_scale_log2,
    int pad_head
) {
    using Types = SparseAttnTypes<kBlockM, kBlockN, kHeadDim, kNWarps, Element>;
    using SmemLayoutQ = typename Types::SmemLayoutQ;
    using SmemLayoutK = typename Types::SmemLayoutK;
    using SmemCopyAtom = typename Types::SmemCopyAtom;
    using GmemTiledCopyQKV = typename Types::GmemTiledCopyQKV;
    using TiledMma = typename Types::TiledMma;
    static constexpr int kNRows = Types::kNRows;

    using BSTypes = BlockScoreTypes<kBlockM, kBlockN, kHeadDim, kNWarps, Element>;
    using SharedStorage = typename BSTypes::SharedStorage;

    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const q_tile_idx = blockIdx.x;
    int const group_idx = blockIdx.y;
    int const bh_idx = blockIdx.z;
    int const thread_idx = threadIdx.x;

    int const sb_start = group_idx * sb_per_group;
    int const sb_end = min(sb_start + sb_per_group, M);

    int const q_start = q_tile_idx * kBlockM;
    if (q_start >= N) return;
    int const actual_rows = min(kBlockM, N - q_start);

    int q_load_start = q_start;
    if (q_load_start + kBlockM > N_phys) q_load_start = N_phys - kBlockM;
    int const q_row_offset = q_start - q_load_start;

    Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.data()), SmemLayoutK{});

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);

    int const q_offset = bh_idx * stride_qk_bh + q_load_start * stride_qk_n;
    Tensor mQ = make_tensor(
        make_gmem_ptr(q_ptr + q_offset),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(stride_qk_n, _1{}));
    {
        Tensor tQgQ = gmem_thr_copy.partition_S(mQ);
        Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
        cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    }
    cute::cp_async_fence();
    sparse_cp_async_wait<0>();
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    {
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    auto taccScO = thr_mma.partition_C(
        cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}));
    auto taccScO_rowcol = make_tensor(taccScO.data(),
        SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(taccScO.layout()));

    int const num_kv_tiles = (score_block_size + kBlockN - 1) / kBlockN;

    auto taccOcO = thr_mma.partition_C(
        cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    auto taccOcO_rowcol = make_tensor(taccOcO.data(),
        SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(taccOcO.layout()));
    auto taccOcO_row = taccOcO_rowcol(_, _0{});

    int64_t const bh_out_base = (int64_t)bh_idx * N * M;

    auto load_K = [&](int kv_start) {
        int actual_start_k = kv_start;
        if (actual_start_k < 0) actual_start_k = 0;
        if (actual_start_k + kBlockN > N_phys) actual_start_k = N_phys - kBlockN;
        int k_offset = bh_idx * stride_qk_bh + actual_start_k * stride_qk_n;
        Tensor gK = make_tensor(
            make_gmem_ptr(k_ptr + k_offset),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            make_stride(stride_qk_n, _1{}));
        Tensor tKgK = gmem_thr_copy.partition_S(gK);
        Tensor tKsK = gmem_thr_copy.partition_D(sK);
        cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
        cute::cp_async_fence();
    };

    for (int sb = sb_start; sb < sb_end; ++sb) {
        int const score_block_start = sb * score_block_size - pad_head;

        SparseSoftmax<kNRows> softmax(softmax_scale_log2);

        load_K(score_block_start);

        for (int tile = 0; tile < num_kv_tiles; ++tile) {
            int const intended_start = score_block_start + tile * kBlockN;
            int actual_start_k = intended_start;
            if (actual_start_k < 0) actual_start_k = 0;
            if (actual_start_k + kBlockN > N_phys) actual_start_k = N_phys - kBlockN;
            int const col_offset = intended_start - actual_start_k;
            int const valid_cols = min(kBlockN, score_block_size - tile * kBlockN);

            Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(tSrS);

            sparse_cp_async_wait<0>();
            __syncthreads();

            Tensor tSrQ_cur = tSrQ;
            Tensor tSrK = thr_mma.partition_fragment_B(sK);
            sparse_gemm_sm80<true>(
                tSrS, tSrQ_cur, tSrK, tSsQ, tSsK,
                tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K, nullptr);

            __syncthreads();
            if (tile + 1 < num_kv_tiles) {
                load_K(score_block_start + (tile + 1) * kBlockN);
            }

            bool need_col_mask = (col_offset > 0 || valid_cols < kBlockN
                                  || intended_start < 0
                                  || intended_start + valid_cols > N);
            if (need_col_mask) {
                Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                    SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(tSrS.layout()));
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

            if (actual_rows < kBlockM) {
                Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                    SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(tSrS.layout()));
                #pragma unroll
                for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                    int row = get<0>(taccScO_rowcol(mi, _0{}));
                    if (row < q_row_offset || row >= q_row_offset + actual_rows) {
                        #pragma unroll
                        for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                            tSrS_rowcol(mi, ni) = -INFINITY;
                        }
                    }
                }
            }

            if (tile == 0) {
                softmax.template max_get_scale<true, true>(tSrS);
                softmax.template online_softmax<true, true>(tSrS);
            } else {
                auto scores_scale = softmax.template max_get_scale<false, false>(tSrS);
                softmax.template online_softmax<false, false>(tSrS);
                (void)scores_scale;
            }
        }

        {
            SparseSumOp sum_op;
            sparse_quad_allreduce(softmax.row_sum, softmax.row_sum, sum_op);
        }

        #pragma unroll
        for (int mi = 0; mi < size(taccOcO_row); ++mi) {
            int tile_row = get<0>(taccOcO_row(mi));
            if (get<1>(taccOcO_row(_0{})) == 0
                && tile_row >= q_row_offset
                && tile_row < q_row_offset + actual_rows) {
                int global_row = q_start + (tile_row - q_row_offset);
                int64_t out_idx = bh_out_base + (int64_t)global_row * M + sb;
                m_locals[out_idx] = softmax.row_max(mi) * softmax_scale_log2;
                l_locals[out_idx] = softmax.row_sum(mi);
            }
        }
    }

    if (num_groups == 1) {
        #pragma unroll
        for (int mi = 0; mi < size(taccOcO_row); ++mi) {
            int tile_row = get<0>(taccOcO_row(mi));
            if (get<1>(taccOcO_row(_0{})) == 0
                && tile_row >= q_row_offset
                && tile_row < q_row_offset + actual_rows) {
                int global_row = q_start + (tile_row - q_row_offset);
                int64_t out_row_base = bh_out_base + (int64_t)global_row * M;

                float m_global = -INFINITY;
                float L_global = 0.0f;
                for (int j = 0; j < M; ++j) {
                    float m_i = m_locals[out_row_base + j];
                    float l_i = l_locals[out_row_base + j];
                    float m_new = fmaxf(m_global, m_i);
                    L_global = L_global * exp2f(m_global - m_new)
                             + l_i * exp2f(m_i - m_new);
                    m_global = m_new;
                }
                float safe_L = (L_global > 0.0f) ? L_global : 1.0f;
                for (int j = 0; j < M; ++j) {
                    float m_i = m_locals[out_row_base + j];
                    float l_i = l_locals[out_row_base + j];
                    block_scores[out_row_base + j] = l_i * exp2f(m_i - m_global) / safe_L;
                }
            }
        }
    }
}


template <int BLOCK_SIZE, int TILE_M>
__global__ void block_normalize_kernel(
    const float* __restrict__ m_locals,
    const float* __restrict__ l_locals,
    float* __restrict__ block_scores,
    int N, int M, int num_bh
) {
    int const bh = blockIdx.z;
    if (bh >= num_bh) return;

    int const query_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (query_idx >= N) return;

    int64_t const row_base = (int64_t)bh * N * M + (int64_t)query_idx * M;
    const float* m_row = m_locals + row_base;
    const float* l_row = l_locals + row_base;

    float m_global = -INFINITY;
    float L_global = 0.0f;

    for (int col = 0; col < M; col += TILE_M) {
        float m_batch = -INFINITY;
        float l_batch = 0.0f;

        #pragma unroll
        for (int j = 0; j < TILE_M; j++) {
            int col_idx = col + j;
            if (col_idx < M) {
                float m_i = __ldg(m_row + col_idx);
                float l_i = __ldg(l_row + col_idx);
                float new_max = fmaxf(m_batch, m_i);
                l_batch = l_batch * exp2f(m_batch - new_max)
                        + l_i * exp2f(m_i - new_max);
                m_batch = new_max;
            }
        }

        float m_new = fmaxf(m_global, m_batch);
        float alpha = exp2f(m_global - m_new);
        float gamma = exp2f(m_batch - m_new);
        L_global = L_global * alpha + l_batch * gamma;
        m_global = m_new;
    }

    float safe_L = (L_global > 0.0f) ? L_global : 1.0f;

    float* out_row = block_scores + row_base;
    for (int col_idx = 0; col_idx < M; col_idx++) {
        float m_i = __ldg(m_row + col_idx);
        float l_i = __ldg(l_row + col_idx);
        float score = l_i * exp2f(m_i - m_global) / safe_L;
        out_row[col_idx] = score;
    }
}


static int cdiv(int a, int b) { return (a + b - 1) / b; }


template <int kBlockM, int kNWarps>
void block_score_dispatch(
    torch::Tensor q, torch::Tensor k,
    torch::Tensor m_locals, torch::Tensor l_locals,
    torch::Tensor block_scores,
    int B, int N, int N_phys, int H, int D,
    int M, int score_block_size, int pad_head,
    float softmax_scale_log2,
    bool is_bf16, cudaStream_t stream
) {
    int stride_qk_bh = N_phys * D;
    int stride_qk_n = D;
    int num_bh = B * H;
    int q_tiles = cdiv(N, kBlockM);

    int sb_per_group = M;
    int num_groups = 1;

    int num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int target_grid = num_sm * 4;
    int grid_total = q_tiles * num_bh;

    if (grid_total < target_grid && M > 1) {
        int needed_groups = cdiv(target_grid, q_tiles * num_bh);
        if (needed_groups > M) needed_groups = M;
        sb_per_group = cdiv(M, needed_groups);
        if (sb_per_group < 1) sb_per_group = 1;
        num_groups = cdiv(M, sb_per_group);
    }

    dim3 grid_a(q_tiles, num_groups, num_bh);
    dim3 block_a(kNWarps * 32);

    #define LAUNCH(BN, HEAD_DIM, Elem) \
        do { \
            using BST = BlockScoreTypes<kBlockM, BN, HEAD_DIM, kNWarps, Elem>; \
            int smem_size = (int)sizeof(typename BST::SharedStorage); \
            auto kernel = block_score_kernel<kBlockM, BN, HEAD_DIM, kNWarps, Elem>; \
            if (smem_size > 48 * 1024) { \
                cudaFuncSetAttribute(kernel, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
            } \
            kernel<<<grid_a, block_a, smem_size, stream>>>( \
                reinterpret_cast<const Elem*>(q.data_ptr()), \
                reinterpret_cast<const Elem*>(k.data_ptr()), \
                m_locals.data_ptr<float>(), \
                l_locals.data_ptr<float>(), \
                block_scores.data_ptr<float>(), \
                N, N_phys, M, score_block_size, \
                sb_per_group, num_groups, \
                stride_qk_bh, stride_qk_n, \
                softmax_scale_log2, pad_head); \
        } while(0)

    #define DISPATCH_DTYPE(BN, HEAD_DIM) \
        do { \
            if (is_bf16) { LAUNCH(BN, HEAD_DIM, cutlass::bfloat16_t); } \
            else         { LAUNCH(BN, HEAD_DIM, cutlass::half_t); } \
        } while(0)

    #define DISPATCH_HEAD_DIM(BN) \
        do { \
            if      (D == 32)  DISPATCH_DTYPE(BN, 32); \
            else if (D == 64)  DISPATCH_DTYPE(BN, 64); \
            else if (D == 96)  DISPATCH_DTYPE(BN, 96); \
            else if (D == 128) DISPATCH_DTYPE(BN, 128); \
            else if (D == 160) DISPATCH_DTYPE(BN, 160); \
            else if (D == 256) DISPATCH_DTYPE(BN, 256); \
        } while(0)

    if (score_block_size >= 128) {
        DISPATCH_HEAD_DIM(64);
    } else {
        DISPATCH_HEAD_DIM(32);
    }

    #undef LAUNCH
    #undef DISPATCH_DTYPE
    #undef DISPATCH_HEAD_DIM

    if (num_groups > 1) {
        constexpr int BLOCK_SIZE_B = 256;
        constexpr int TILE_M_B = 16;
        dim3 grid_b(cdiv(N, BLOCK_SIZE_B), 1, num_bh);
        dim3 block_b(BLOCK_SIZE_B);
        block_normalize_kernel<BLOCK_SIZE_B, TILE_M_B><<<grid_b, block_b, 0, stream>>>(
            m_locals.data_ptr<float>(),
            l_locals.data_ptr<float>(),
            block_scores.data_ptr<float>(),
            N, M, num_bh);
    }
}


void block_score_launch(
    torch::Tensor q, torch::Tensor k,
    torch::Tensor m_locals, torch::Tensor l_locals,
    torch::Tensor block_scores,
    int B, int N, int N_phys, int H, int D,
    int M, int score_block_size,
    float softmax_scale,
    int pad_head
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    float softmax_scale_log2 = softmax_scale * float(M_LOG2E);
    bool is_bf16 = (q.scalar_type() == at::ScalarType::BFloat16);

    if (D <= 64) {
        block_score_dispatch<32, 2>(
            q, k, m_locals, l_locals, block_scores,
            B, N, N_phys, H, D, M, score_block_size, pad_head,
            softmax_scale_log2, is_bf16, stream);
    } else {
        block_score_dispatch<64, 4>(
            q, k, m_locals, l_locals, block_scores,
            B, N, N_phys, H, D, M, score_block_size, pad_head,
            softmax_scale_log2, is_bf16, stream);
    }
}
