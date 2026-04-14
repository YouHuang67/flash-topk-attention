#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>


template <int SORT_SIZE, int ITEMS_PER_THREAD>
__global__ void qblock_merge_kernel(
    const int32_t* __restrict__ topk_indices,
    const float* __restrict__ topk_scores,
    int32_t* __restrict__ merged_indices,
    float* __restrict__ merged_scores,
    int q_block_size, int M_OUT, int M_TOTAL,
    int QBLOCK_TOPK, int num_slices
) {
    constexpr int BLOCK_SIZE = 128;
    static_assert(SORT_SIZE == BLOCK_SIZE * ITEMS_PER_THREAD, "");

    using BlockSort   = cub::BlockRadixSort<float, BLOCK_SIZE,
                                            ITEMS_PER_THREAD, int32_t>;
    using BlockStoreF = cub::BlockStore<float, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_STORE_TRANSPOSE>;
    using BlockStoreI = cub::BlockStore<int32_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_STORE_TRANSPOSE>;

    __shared__ union {
        float                              scatter_buf[SORT_SIZE];
        typename BlockSort::TempStorage    sort;
        typename BlockStoreF::TempStorage  store_f;
        typename BlockStoreI::TempStorage  store_i;
    } smem;

    int slice = blockIdx.x;
    if (slice >= num_slices) return;

    for (int i = threadIdx.x; i < SORT_SIZE; i += BLOCK_SIZE) {
        smem.scatter_buf[i] = 0.0f;
    }
    __syncthreads();

    int num_pairs = q_block_size * M_OUT;
    int64_t slice_input_base = (int64_t)slice * q_block_size * M_OUT;

    for (int pair_idx = threadIdx.x; pair_idx < num_pairs; pair_idx += BLOCK_SIZE) {
        int64_t global_offset = slice_input_base + pair_idx;
        int32_t idx = topk_indices[global_offset];
        float score = topk_scores[global_offset];

        if (idx >= 0 && idx < M_TOTAL) {
            atomicAdd(&smem.scatter_buf[idx], score);
        }
    }
    __syncthreads();

    float keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int pos = threadIdx.x * ITEMS_PER_THREAD + i;
        if (pos < M_TOTAL) {
            keys[i] = smem.scatter_buf[pos];
            vals[i] = pos;
        } else {
            keys[i] = -1.0f;
            vals[i] = -1;
        }
    }
    __syncthreads();

    BlockSort(smem.sort).SortDescending(keys, vals);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int pos = threadIdx.x * ITEMS_PER_THREAD + i;
        if (pos >= QBLOCK_TOPK || keys[i] <= 0.0f) {
            keys[i] = 0.0f;
            vals[i] = -1;
        }
    }

    int64_t out_offset = (int64_t)slice * QBLOCK_TOPK;
    BlockStoreF(smem.store_f).Store(
        merged_scores + out_offset, keys, QBLOCK_TOPK);
    __syncthreads();
    BlockStoreI(smem.store_i).Store(
        merged_indices + out_offset, vals, QBLOCK_TOPK);
}


static unsigned int next_pow2(unsigned int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}


void qblock_merge_launch(
    torch::Tensor topk_indices,
    torch::Tensor topk_scores,
    torch::Tensor merged_indices,
    torch::Tensor merged_scores,
    int q_block_size, int M_OUT, int M_TOTAL,
    int QBLOCK_TOPK, int num_slices
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int32_t* idx_ptr = topk_indices.data_ptr<int32_t>();
    const float* scr_ptr = topk_scores.data_ptr<float>();
    int32_t* out_idx_ptr = merged_indices.data_ptr<int32_t>();
    float* out_scr_ptr = merged_scores.data_ptr<float>();

    unsigned int sort_size = next_pow2((unsigned int)M_TOTAL);
    if (sort_size < 128) sort_size = 128;

    #define LAUNCH(SS)                                                      \
        qblock_merge_kernel<SS, (SS)/128><<<num_slices, 128, 0, stream>>>(  \
            idx_ptr, scr_ptr, out_idx_ptr, out_scr_ptr,                     \
            q_block_size, M_OUT, M_TOTAL, QBLOCK_TOPK, num_slices)

    switch (sort_size) {
        case 4096: LAUNCH(4096); break;
        case 2048: LAUNCH(2048); break;
        case 1024: LAUNCH(1024); break;
        case  512: LAUNCH( 512); break;
        case  256: LAUNCH( 256); break;
        default:   LAUNCH( 128); break;
    }
    #undef LAUNCH
}
