/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>

#include "backend/habana_device/HPUAllocator.h"
#include "backend/helpers/tensor_utils.h"
#include "common/dump_args.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "hpu_ops/op_logger.h"
#include "instance_norm.h"

namespace habana {
namespace eager {

namespace {
constexpr size_t INPUT_BATCH_INDEX = 0;
constexpr size_t INPUT_CHANNEL_INDEX = 1;
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_fwd_eager_hpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
  PT_OP_INFO("instance_norm_eager: ", DUMP_4ARGS(input, weight, bias, eps));

  auto InstanceNormMeta = [](const at::Stack& stack) {
    const auto& input = stack.at(0).toTensor();
    OutputMetaDataVector meta(3);
    meta.at(0).shape = input.sizes().vec();
    meta.at(1).shape = {
        input.sizes().vec()[INPUT_BATCH_INDEX],
        input.sizes().vec()[INPUT_CHANNEL_INDEX]};
    meta.at(2).shape = {
        input.sizes().vec()[INPUT_BATCH_INDEX],
        input.sizes().vec()[INPUT_CHANNEL_INDEX]};
    meta.at(0).dtype = input.scalar_type();
    meta.at(1).dtype = c10::ScalarType::Float;
    meta.at(2).dtype = c10::ScalarType::Float;
    return meta;
  };

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::instance_norm", {input, weight, bias, eps}};
  hpu_op.SetOutputMetaFn(InstanceNormMeta);

  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_bwd_eager_hpu(
    const at::Tensor& input,
    const at::Tensor& grad_in,
    const at::Tensor& mean,
    const at::Tensor& istd,
    const at::Tensor& gamma) {
  PT_OP_INFO(
      "instance_norm_backward_eager: ",
      DUMP_5ARGS(input, grad_in, mean, istd, gamma));

  auto InstanceNormBackwardMeta = [](const at::Stack& stack) {
    const auto& input = stack.at(0).toTensor();
    OutputMetaDataVector meta(3);
    meta.at(0).shape = input.sizes().vec();
    meta.at(1).shape = {input.sizes().vec()[INPUT_CHANNEL_INDEX]};
    meta.at(2).shape = {input.sizes().vec()[INPUT_CHANNEL_INDEX]};
    meta.at(0).dtype = input.scalar_type();
    meta.at(1).dtype = c10::ScalarType::Float;
    meta.at(2).dtype = c10::ScalarType::Float;
    return meta;
  };
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::instance_norm_backward", {input, grad_in, mean, istd, gamma}};

  hpu_op.SetOutputMetaFn(InstanceNormBackwardMeta);
  return hpu_op.call();
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("instance_norm", instance_norm_fwd_eager_hpu);
  m.impl("instance_norm_backward", instance_norm_bwd_eager_hpu);
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "instance_norm(Tensor input, Tensor weight, Tensor bias, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "instance_norm_backward(Tensor input, Tensor grad_in, Tensor mean, Tensor istd, Tensor gamma) -> (Tensor, Tensor, Tensor)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
dispatch_instance_norm_backward_hpu(
    const at::Tensor& input,
    const at::Tensor& grad_in,
    const at::Tensor& mean,
    const at::Tensor& istd,
    const at::Tensor& gamma) {
  PT_OP_INFO(
      "Dispatch hpu::instance_norm_backward: ",
      DUMP_5ARGS(input, grad_in, mean, istd, gamma));
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::instance_norm_backward", "")
                       .typed<decltype(dispatch_instance_norm_backward_hpu)>();
  return op.call(input, grad_in, mean, istd, gamma);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_instance_norm_hpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
  PT_OP_INFO(
      "Dispatch hpu::instance_norm: ", DUMP_4ARGS(input, weight, bias, eps));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::instance_norm", "")
                       .typed<decltype(dispatch_instance_norm_hpu)>();
  return op.call(input, weight, bias, eps);
}

class InstanceNormAutogradHPU
    : public torch::autograd::Function<InstanceNormAutogradHPU> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight, // gamma
      const at::Tensor& bias, // beta
      double eps) {
    auto input_maybe_reshaped = input;
    const auto is_3d = input.dim() == 3;
    if (is_3d) {
      auto new_shape = input.sizes().vec();
      new_shape.push_back(1);
      input_maybe_reshaped = at::reshape(input, new_shape);
    }

    at::Tensor output;

    auto [output_maybe_reshaped, mean, istd] =
        dispatch_instance_norm_hpu(input, weight, bias, eps);

    if (is_3d) {
      output = at::reshape(output_maybe_reshaped, input.sizes());
    } else {
      output = output_maybe_reshaped;
    }

    ctx->save_for_backward({input, mean, istd, weight});
    return output;
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<at::Tensor> grad_in) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto mean = saved[1];
    auto istd = saved[2];
    auto gamma = saved[3];

    auto input_maybe_reshaped = input;
    auto grad_in_maybe_reshaped = grad_in[0];
    const auto is_3d = input.dim() == 3;
    if (is_3d) {
      auto new_shape = input.sizes().vec();
      new_shape.push_back(1);
      input_maybe_reshaped = at::reshape(input, new_shape);
      grad_in_maybe_reshaped = at::reshape(grad_in[0], new_shape);
    }

    at::Tensor grad_out;

    auto [grad_out_maybe_reshaped, grad_beta, grad_gamma] =
        dispatch_instance_norm_backward_hpu(
            input_maybe_reshaped, grad_in_maybe_reshaped, mean, istd, gamma);

    if (is_3d) {
      grad_out = at::reshape(grad_out_maybe_reshaped, input.sizes());
    } else {
      grad_out = grad_out_maybe_reshaped;
    }

    return {grad_out, grad_gamma, grad_beta, at::Tensor()};
  }
};

at::Tensor instance_norm_autograd_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool use_input_stats,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  auto weight =
      weight_opt.value_or(at::ones(input.sizes().vec()[INPUT_CHANNEL_INDEX])
                              .to(torch::kFloat32)
                              .to(torch::kHPU));
  auto bias =
      bias_opt.value_or(at::zeros(input.sizes().vec()[INPUT_CHANNEL_INDEX])
                            .to(torch::kFloat32)
                            .to(torch::kHPU));

  PT_OP_INFO(
      " instance_norm:",
      DUMP_9ARGS(
          input,
          weight,
          bias,
          running_mean_opt,
          running_var_opt,
          use_input_stats,
          momentum,
          eps,
          cudnn_enabled));

  return InstanceNormAutogradHPU::apply(input, weight, bias, eps);
}

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  m.impl("instance_norm", instance_norm_autograd_wrap);
}

} // namespace eager
} // namespace habana
