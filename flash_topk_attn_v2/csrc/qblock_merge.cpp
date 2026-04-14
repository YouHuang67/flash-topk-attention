#include <torch/extension.h>

void qblock_merge_launch(
    torch::Tensor topk_indices,
    torch::Tensor topk_scores,
    torch::Tensor merged_indices,
    torch::Tensor merged_scores,
    int q_block_size, int M_OUT, int M_TOTAL,
    int QBLOCK_TOPK, int num_slices
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qblock_merge_launch", &qblock_merge_launch);
}
