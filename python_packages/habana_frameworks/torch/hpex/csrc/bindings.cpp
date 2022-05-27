#include <torch/extension.h>

#include "bindings.h"
#include "habana_kernels/wrap_kernels_declarations.h"

// Wrappers to match singatures
static void optimizer_fused_adamw(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const float lr,
    at::Tensor& neg_step,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay) {
  at::TensorList gradients(gradient_vec);
  at::TensorList weights(weight_vec);
  at::TensorList exp_avg(exp_avg_vec);
  at::TensorList exp_avg_sq(exp_avg_sq_vec);

  optimizer_adamw_hpu_wrap(
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      lr,
      neg_step,
      beta1,
      beta2,
      epsilon,
      weight_decay);
}

static void optimizer_fused_adagrad(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& variance_vec,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  at::TensorList gradients(gradient_vec);
  at::TensorList weights(weight_vec);
  at::TensorList variances(variance_vec);

  optimizer_adagrad_hpu_wrap(
      gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
}

static void optimizer_fused_ema(
    const std::vector<at::Tensor>& model_inputs,
    std::vector<at::Tensor>& updated_ema,
    const at::Tensor& decay) {
  at::TensorList modelInputs(model_inputs);
  at::TensorList updatedEma(updated_ema);
  optimizer_ema_hpu_wrap(modelInputs, updatedEma, decay);
}

static void optimizer_fused_sgd(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  at::TensorList gradients(gradient_vec);
  at::TensorList weights(weight_vec);

  optimizer_sgd_hpu_wrap(gradients, weights, lr, wd, mom, damp, nesterov);
}

static void optimizer_fused_sgd_momentum(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& momentum_vec,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  at::TensorList gradients(gradient_vec);
  at::TensorList weights(weight_vec);
  at::TensorList momentum(momentum_vec);

  optimizer_sgd_momentum_hpu_wrap(
      gradients, weights, momentum, epoch_num, lr, wd, mom, damp, nesterov);
}

static std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count(
    const torch::Tensor& gradients,
    torch::Tensor& weights_in,
    torch::Tensor& moments_in,
    const torch::Tensor& indices,
    const torch::Tensor& learning_rate,
    const torch::Tensor& valid_count) {
  return optimizer_sparse_sgd_with_valid_count_hpu_wrap(
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count,
      0.0f,
      false);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //////////////////////////// Optimizers /////////////////////////////////////
  m.def(
      "fused_adamw",
      &optimizer_fused_adamw,
      "Compute and apply gradient update to parameters for Adam optimizer");
  m.def(
      "fused_lamb_phase1",
      &optimizer_lamb_phase1_hpu_wrap,
      "Compute and apply gradient update to parameters for lamb optimizer phase1");
  m.def(
      "fused_lamb_phase2",
      &optimizer_lamb_phase2_hpu_wrap,
      "Compute and apply gradient update to parameters for lamb optimizer phase2");
  m.def(
      "fused_lamb_norm",
      &optimizer_lamb_fused_norm_hpu_wrap,
      "Compute and apply global grad norm for lamb optimizer");
  m.def(
      "fused_adagrad",
      &optimizer_fused_adagrad,
      "Compute and apply gradient update to parameters for Adagrad optimizer");
  m.def(
      "fused_sgd",
      &optimizer_fused_sgd,
      "Compute and apply gradient update to parameters for SGD optimizer");
  m.def(
      "fused_ema",
      &optimizer_fused_ema,
      "Compute and apply exponential moving avg update in ema optimizer");
  m.def(
      "fused_sgd_momentum",
      &optimizer_fused_sgd_momentum,
      "Compute and apply gradient update to parameters for SGD with momentum optimizer");
  m.def(
      "sparse_sgd_with_valid_count",
      &optimizer_sparse_sgd_with_valid_count,
      "Optimizer Sparse Stochastic Gradient Descent with valid count");
  m.def(
      "sparse_adagrad_with_valid_count",
      &optimizer_sparse_adagrad_with_valid_count_hpu_wrap,
      "Optimizer Sparse Adagrad with valid count ");

  //////////////////////////// Normalizations /////////////////////////////////
  m.def(
      "fused_norm",
      &fused_norm_hpu_wrap,
      "Compute the norm of the norm of the input vector of tensors");

  //////////////////////////// Kernels ////////////////////////////////////////
  m.def(
      "custom_nms",
      &torchvision_nms_hpu_wrap,
      "NMS operation for boxes of a single class");
  m.def(
      "batched_nms",
      &batched_nms_hpu_wrap,
      "NMS operation for boxes of a multiple classes");
  m.def(
      "embedding_bag_sum_fwd",
      &embedding_bag_sum_hpu_wrap,
      "embedding bag sum forward");
  m.def(
      "embedding_bag_sum_bwd",
      &embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap,
      "embedding bag sum bwd");
  m.def(
      "embedding_bag_preproc", &embedding_bag_preproc, "embedding bag preproc");
  m.def(
      "roi_align_forward",
      &vision::ops::roi_align_fwd_wrap,
      "ROI Align forward");
  m.def(
      "roi_align_backward",
      &vision::ops::roi_align_bwd_wrap,
      "ROI Align backward");
}
