#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <iostream>

// Fused AdamW
// Input tensors
// 1     Gradient             FP32/FP16/BF16  1D/2D
// 2     Weight               FP32/FP16/BF16  1D/2D
// 3     Exp_avg              FP32/FP16/BF16  1D/2D
// 4     Exp_avg_sq           FP32/FP16/BF16  1D/2D
// 5     Learning Rate
// 6     Beta1
// 7     Beta2
// 8     Epsilon
// 9     Step
// 10    Bias_correction
// 11    Float weight_decay
//
// Output
// None
extern void optimizer_adamw_hpu_wrap(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay);

void optimizer_fused_adamw(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay) {
  optimizer_adamw_hpu_wrap(
      gradient_vec,
      weight_vec,
      exp_avg_vec,
      exp_avg_sq_vec,
      lr,
      beta1,
      beta2,
      epsilon,
      step,
      bias_correction,
      weight_decay);
}

// Fused norm
// Input tensors
// 1     Gradient             FP32/FP16/BF16  1D/2D
//
// Output
// 1     Norm
extern at::Tensor fused_norm_hpu_wrap(
    const std::vector<at::Tensor>& grad,
    float norm_type);

at::Tensor custom_fused_norm(
    const std::vector<at::Tensor>& grad,
    float norm_type) {
  return fused_norm_hpu_wrap(grad, norm_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fused_adamw",
      &optimizer_fused_adamw,
      "Compute and apply gradient update to parameters for Adam optimizer");
  m.def(
      "fused_norm",
      &custom_fused_norm,
      "Compute the norm of the norm of the input vector of tensors");
}
