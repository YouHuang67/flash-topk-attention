#include <torch/extension.h>

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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attn_bwd_dq_launch", &sparse_attn_bwd_dq_launch);
}
