#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>


__global__ void build_reverse_indices_kernel(
    const int32_t* __restrict__ merged_indices,
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ reverse_indices,
    int32_t* __restrict__ reverse_counts,
    int QM, int S_MAX, int M
) {
    int qm = blockIdx.x;
    int bh = blockIdx.z;

    int cnt = counts[bh * QM + qm];
    int base = (bh * QM + qm) * S_MAX;

    for (int s = threadIdx.x; s < cnt; s += blockDim.x) {
        int kv_block = merged_indices[base + s];
        if (kv_block >= 0 && kv_block < M) {
            int slot = atomicAdd(&reverse_counts[bh * M + kv_block], 1);
            reverse_indices[(bh * M + kv_block) * QM + slot] = qm;
        }
    }
}


template <int BLOCK_DIM, int ITEMS_PER_THREAD>
__global__ void argsort_reverse_counts_kernel(
    const int32_t* __restrict__ reverse_counts,
    int32_t* __restrict__ sorted_kv_indices,
    int M
) {
    int bh = blockIdx.x;
    const int32_t* counts_bh = reverse_counts + bh * M;
    int32_t* out_bh = sorted_kv_indices + bh * M;

    using BlockRadixSort = cub::BlockRadixSort<
        int32_t, BLOCK_DIM, ITEMS_PER_THREAD, int32_t>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    int32_t keys[ITEMS_PER_THREAD];
    int32_t vals[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x * ITEMS_PER_THREAD + i;
        keys[i] = (idx < M) ? counts_bh[idx] : 0;
        vals[i] = idx;
    }

    BlockRadixSort(temp_storage).SortDescending(keys, vals);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x * ITEMS_PER_THREAD + i;
        if (idx < M) {
            out_bh[idx] = vals[i];
        }
    }
}


#define LAUNCH_ARGSORT(BD, IPT) \
    argsort_reverse_counts_kernel<BD, IPT><<<BH, BD, 0, stream>>>( \
        reverse_counts.data_ptr<int32_t>(), \
        sorted_kv_indices.data_ptr<int32_t>(), M)


void build_reverse_indices_launch(
    torch::Tensor merged_indices,
    torch::Tensor counts,
    torch::Tensor reverse_indices,
    torch::Tensor reverse_counts,
    torch::Tensor sorted_kv_indices,
    int B, int H, int QM, int S_MAX, int M
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int BH = B * H;

    {
        constexpr int kThreads = 128;
        dim3 grid(QM, 1, BH);
        build_reverse_indices_kernel<<<grid, kThreads, 0, stream>>>(
            merged_indices.data_ptr<int32_t>(),
            counts.data_ptr<int32_t>(),
            reverse_indices.data_ptr<int32_t>(),
            reverse_counts.data_ptr<int32_t>(),
            QM, S_MAX, M);
    }

    {
        if (M <= 128) {
            LAUNCH_ARGSORT(128, 1);
        } else if (M <= 256) {
            LAUNCH_ARGSORT(128, 2);
        } else if (M <= 512) {
            LAUNCH_ARGSORT(128, 4);
        } else {
            LAUNCH_ARGSORT(256, 4);
        }
    }
}
