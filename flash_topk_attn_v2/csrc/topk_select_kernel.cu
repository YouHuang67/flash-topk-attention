#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

template <int SORT_SIZE, int ITEMS_PER_THREAD>
__global__ void topk_select_kernel(
    const float* __restrict__ block_scores,
    int32_t* __restrict__ topk_indices,
    float* __restrict__ topk_raw_scores,
    float* __restrict__ topk_avg_scores,
    int M, int M_OUT,
    int score_block_size, int pad_head, int pad_tail,
    float threshold, int num_slices
) {
    constexpr int BLOCK_SIZE = SORT_SIZE / ITEMS_PER_THREAD;

    using BlockLoadF  = cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD,
                                       cub::BLOCK_LOAD_TRANSPOSE>;
    using BlockSort   = cub::BlockRadixSort<float, BLOCK_SIZE,
                                            ITEMS_PER_THREAD, int32_t>;
    using BlockScan   = cub::BlockScan<float, BLOCK_SIZE>;
    using BlockReduce = cub::BlockReduce<int, BLOCK_SIZE>;
    using BlockStoreF = cub::BlockStore<float, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_STORE_TRANSPOSE>;
    using BlockStoreI = cub::BlockStore<int32_t, BLOCK_SIZE, ITEMS_PER_THREAD,
                                        cub::BLOCK_STORE_TRANSPOSE>;

    __shared__ union {
        typename BlockLoadF::TempStorage  load;
        typename BlockSort::TempStorage   sort;
        typename BlockScan::TempStorage   scan;
        typename BlockReduce::TempStorage reduce;
        typename BlockStoreF::TempStorage store_f;
        typename BlockStoreI::TempStorage store_i;
    } smem;
    __shared__ int smem_cut;

    int slice = blockIdx.x;
    if (slice >= num_slices) return;

    float keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];

    BlockLoadF(smem.load).Load(
        block_scores + (int64_t)slice * M, keys, M, -CUDART_INF_F);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int sorted_pos = threadIdx.x * ITEMS_PER_THREAD + i;
        vals[i] = sorted_pos;
        if (sorted_pos < M) {
            int valid_count = score_block_size;
            if (pad_head > 0 && sorted_pos == 0) valid_count -= pad_head;
            if (pad_tail > 0 && sorted_pos == M - 1) valid_count -= pad_tail;
            keys[i] /= (float)valid_count;
        }
    }

    BlockSort(smem.sort).SortDescending(keys, vals);
    __syncthreads();

    float local_raw[ITEMS_PER_THREAD];
    float local_cum[ITEMS_PER_THREAD];
    float local_sum = 0.0f;
    constexpr int CUMSUM_SHIFT = 12;
    float scaled_threshold = ldexpf(threshold, CUMSUM_SHIFT);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int sorted_pos = threadIdx.x * ITEMS_PER_THREAD + i;
        float raw = 0.0f;
        if (sorted_pos < M) {
            int orig_idx = vals[i];
            int valid_count = score_block_size;
            if (pad_head > 0 && orig_idx == 0) valid_count -= pad_head;
            if (pad_tail > 0 && orig_idx == M - 1) valid_count -= pad_tail;
            raw = keys[i] * (float)valid_count;
        }
        local_raw[i] = raw;
        local_sum += ldexpf(raw, CUMSUM_SHIFT);
        local_cum[i] = local_sum;
    }

    float prefix;
    BlockScan(smem.scan).ExclusiveSum(local_sum, prefix);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
        local_cum[i] += prefix;

    int thread_cut = SORT_SIZE;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int sorted_pos = threadIdx.x * ITEMS_PER_THREAD + i;
        if (sorted_pos < M && local_cum[i] >= scaled_threshold && thread_cut == SORT_SIZE)
            thread_cut = sorted_pos;
    }

    int block_cut = BlockReduce(smem.reduce).Reduce(thread_cut, cub::Min());
    __syncthreads();
    if (threadIdx.x == 0) {
        int cut = (block_cut < SORT_SIZE) ? block_cut + 1 : M;
        smem_cut = min(cut, M_OUT);
    }
    __syncthreads();
    int cut = smem_cut;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int sorted_pos = threadIdx.x * ITEMS_PER_THREAD + i;
        if (sorted_pos >= cut || sorted_pos >= M_OUT) {
            keys[i] = 0.0f;
            vals[i] = -1;
            local_raw[i] = 0.0f;
        }
    }

    int64_t out_offset = (int64_t)slice * M_OUT;
    BlockStoreF(smem.store_f).Store(
        topk_avg_scores + out_offset, keys, M_OUT);
    __syncthreads();
    BlockStoreI(smem.store_i).Store(
        topk_indices + out_offset, vals, M_OUT);
    __syncthreads();
    BlockStoreF(smem.store_f).Store(
        topk_raw_scores + out_offset, local_raw, M_OUT);
}


static unsigned int next_pow2(unsigned int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}


void topk_select_launch(
    torch::Tensor block_scores,
    torch::Tensor topk_indices,
    torch::Tensor topk_raw_scores,
    torch::Tensor topk_avg_scores,
    int M, int M_OUT,
    int score_block_size, int pad_head, int pad_tail,
    float threshold, int num_slices
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const float* in_ptr = block_scores.data_ptr<float>();
    int32_t* idx_ptr = topk_indices.data_ptr<int32_t>();
    float* raw_ptr = topk_raw_scores.data_ptr<float>();
    float* avg_ptr = topk_avg_scores.data_ptr<float>();

    #define LAUNCH(SS, IPT)                                            \
        topk_select_kernel<SS, IPT><<<num_slices, (SS)/(IPT), 0, stream>>>( \
            in_ptr, idx_ptr, raw_ptr, avg_ptr,                         \
            M, M_OUT, score_block_size, pad_head, pad_tail,            \
            threshold, num_slices)

    unsigned int sort_size = next_pow2((unsigned int)M);
    if (sort_size < 32) sort_size = 32;

    switch (sort_size) {
        case 4096: LAUNCH(4096, 32); break;
        case 2048: LAUNCH(2048, 32); break;
        case 1024: LAUNCH(1024, 32); break;
        case  512: LAUNCH( 512, 16); break;
        case  256: LAUNCH( 256,  8); break;
        case  128: LAUNCH( 128,  4); break;
        case   64: LAUNCH(  64,  2); break;
        default:   LAUNCH(  32,  2); break;
    }
    #undef LAUNCH
}
