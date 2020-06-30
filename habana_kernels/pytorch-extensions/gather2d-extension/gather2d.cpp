#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <iostream>

using namespace std;

extern torch::Tensor gather2d_hpu(
    const torch::Tensor& input,
    const torch::Tensor& indices,
    int64_t validCount);

// Input tensor 1	Input feature map	BF16/FP32	2D
// Input tensor 2	Indices Tensor	I32	1D
// Input tensor 3	Valid Count Tensor	I32	1D
// Output tensor 1	Output feature map	FP32	2D
// GUID: gather_with_valid_count_2d_<bf16/ f32>

torch::Tensor gather_with_valid_count_2d_f32(
    torch::Tensor input,
    torch::Tensor indices,
    int64_t validCount) {
  torch::Tensor out;

  std::cout << "Inside New Op :: gather_with_valid_count_2d_f32 " << std::endl;
  out = gather2d_hpu(input, indices, validCount);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gather_with_valid_count_2d_f32, "Gather 2D forward");
  m.def("backward", &gather_with_valid_count_2d_f32, "TO BE REMOVED");
}
