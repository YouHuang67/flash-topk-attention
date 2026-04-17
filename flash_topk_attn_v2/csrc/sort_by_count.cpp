#include <torch/extension.h>

void sort_by_count_launch(
    torch::Tensor reverse_counts,
    torch::Tensor sorted_counts,
    torch::Tensor sorted_global_ids,
    int total,
    int end_bit
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sort_by_count_launch", &sort_by_count_launch);
}
