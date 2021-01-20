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

extern std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
optimizer_lamb_phase1_hpu(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const at::Tensor& clip_global_grad_norm,
    const int grad_averaging,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay);

extern at::Tensor optimizer_lamb_fused_norm_hpu(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm);

void optimizer_lamb_phase2_hpu(
    std::vector<at::Tensor>& weight_vec,
    const std::vector<at::Tensor>& adam_norm_vec,
    const std::vector<at::Tensor>& weight_norm_vec,
    const std::vector<at::Tensor>& adam_step_vec,
    const std::vector<at::Tensor>& trust_ratio_vec,
    const float step,
    const float weight_decay,
    const int use_lamb);

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
    std::vector<at::Tensor>& grad,
    const at::Tensor& max_norm,
    float norm_type);

at::Tensor custom_fused_norm(
    std::vector<at::Tensor>& grad,
    const at::Tensor& max_norm,
    float norm_type) {
  return fused_norm_hpu_wrap(grad, max_norm, norm_type);
}

std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
optimizer_fused_lamb_phase1(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const at::Tensor& clip_global_grad_norm,
    const int grad_averaging,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay) {
  return optimizer_lamb_phase1_hpu(
      gradient_vec,
      weight_vec,
      exp_avg_vec,
      exp_avg_sq_vec,
      clip_global_grad_norm,
      grad_averaging,
      lr,
      beta1,
      beta2,
      epsilon,
      step,
      bias_correction,
      weight_decay);
}

void optimizer_fused_lamb_phase2(
    std::vector<at::Tensor>& weight_vec,
    const std::vector<at::Tensor>& adam_norm_vec,
    const std::vector<at::Tensor>& weight_norm_vec,
    const std::vector<at::Tensor>& adam_step_vec,
    const std::vector<at::Tensor>& trust_ratio_vec,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  optimizer_lamb_phase2_hpu(
      weight_vec,
      adam_norm_vec,
      weight_norm_vec,
      adam_step_vec,
      trust_ratio_vec,
      step,
      weight_decay,
      use_lamb);
}

at::Tensor optimizer_lamb_fused_norm(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
  return optimizer_lamb_fused_norm_hpu(grad, max_grad_norm);
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
  m.def(
      "fused_lamb_phase1",
      &optimizer_fused_lamb_phase1,
      "Compute and apply gradient update to parameters for lamb optimizer phase1");
  m.def(
      "fused_lamb_phase2",
      &optimizer_fused_lamb_phase2,
      "Compute and apply gradient update to parameters for lamb optimizer phase2");
  m.def(
      "fused_lamb_norm",
      &optimizer_lamb_fused_norm,
      "Compute and apply global grad norm for lamb optimizer");
}
