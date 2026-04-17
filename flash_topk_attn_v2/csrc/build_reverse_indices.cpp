#include <torch/extension.h>

void build_reverse_indices_launch(
    torch::Tensor merged_indices,
    torch::Tensor counts,
    torch::Tensor reverse_indices,
    torch::Tensor reverse_counts,
    torch::Tensor sorted_kv_indices,
    int B, int H, int QM, int S_MAX, int M
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_reverse_indices_launch", &build_reverse_indices_launch);
}
