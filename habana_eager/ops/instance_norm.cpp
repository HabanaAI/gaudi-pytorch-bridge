/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "instance_norm.h"
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include "backend/helpers/tensor_utils.h"
#include "common/dump_args.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/instance_norm_utils.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "hpu_ops/op_logger.h"

namespace habana {
namespace eager {

namespace {
constexpr size_t INPUT_BATCH_INDEX = 0;
constexpr size_t INPUT_CHANNEL_INDEX = 1;
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> instance_norm_fwd_eager_hpu(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
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
    const c10::optional<at::Tensor>& gamma_opt) {
  PT_OP_INFO(
      "instance_norm_backward_eager: ",
      DUMP_5ARGS(input, grad_in, mean, istd, gamma_opt));

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
      "hpu::instance_norm_backward", {input, grad_in, mean, istd, gamma_opt}};

  hpu_op.SetOutputMetaFn(InstanceNormBackwardMeta);
  return hpu_op.call();
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("instance_norm", instance_norm_fwd_eager_hpu);
  m.impl("instance_norm_backward", instance_norm_bwd_eager_hpu);
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "instance_norm_backward(Tensor input, Tensor grad_in, Tensor mean, Tensor istd, Tensor? gamma) -> (Tensor, Tensor, Tensor)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
dispatch_instance_norm_backward_hpu(
    const at::Tensor& input,
    const at::Tensor& grad_in,
    const at::Tensor& mean,
    const at::Tensor& istd,
    const c10::optional<at::Tensor>& gamma_opt) {
  PT_OP_INFO(
      "Dispatch hpu::instance_norm_backward: ",
      DUMP_5ARGS(input, grad_in, mean, istd, gamma_opt));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::instance_norm_backward", "")
                       .typed<decltype(dispatch_instance_norm_backward_hpu)>();
  return op.call(input, grad_in, mean, istd, gamma_opt);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_instance_norm_hpu(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double eps) {
  PT_OP_INFO(
      "Dispatch hpu::instance_norm: ",
      DUMP_4ARGS(input, weight_opt, bias_opt, eps));

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::instance_norm", "")
                       .typed<decltype(dispatch_instance_norm_hpu)>();
  return op.call(input, weight_opt, bias_opt, eps);
}

struct InstanceNormBackward
    : public torch::autograd::Function<InstanceNormBackward> {
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& input,
      const c10::optional<at::Tensor>& weight_opt, // gamma
      const torch::Tensor& grad_in,
      const torch::Tensor& mean, // save_mean
      const torch::Tensor& istd, // save_invstd
      const double eps,
      const bool bias_opt) {
    auto input_maybe_reshaped = input;
    auto grad_in_maybe_reshaped = grad_in;
    const auto is_3d = input.dim() == 3;
    if (is_3d) {
      auto new_shape = input.sizes().vec();
      new_shape.push_back(1);
      input_maybe_reshaped = at::reshape(input, new_shape);
      grad_in_maybe_reshaped = at::reshape(grad_in, new_shape);
    }
    const auto [grad_out_maybe_reshaped, grad_beta, grad_gamma] =
        dispatch_instance_norm_backward_hpu(
            input_maybe_reshaped,
            grad_in_maybe_reshaped,
            mean,
            istd,
            weight_opt);

    const torch::Tensor grad_out = is_3d
        ? at::reshape(grad_out_maybe_reshaped, input.sizes())
        : grad_out_maybe_reshaped;
    ctx->save_for_backward(
        {input, weight_opt.value_or(at::Tensor()), grad_in, mean, istd});
    ctx->saved_data["eps"] = eps;
    ctx->saved_data["weight_opt"] = weight_opt.has_value();
    ctx->saved_data["bias_opt"] = bias_opt;
    return {grad_out, grad_gamma, grad_beta};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_in) {
    const auto saved = ctx->get_saved_variables();
    auto input = saved[0]; // Input always as [N, C, X1, ... ,Xn]
    auto weight = saved[1];
    auto gO = saved[2];
    auto save_mean = saved[3];
    auto save_invstd = saved[4];

    const double eps = ctx->saved_data["eps"].toDouble();
    const bool weight_opt = ctx->saved_data["weight_opt"].toBool();
    const bool bias_opt = ctx->saved_data["bias_opt"].toBool();

    const std::array<bool, 3> mask{true, weight_opt, bias_opt};

    const auto input_shape = input.sizes();
    const auto dim0 = input_shape[0];
    const auto dim1 = input_shape[1];
    const auto is_3d = input.dim() == 3;
    const auto is_5d = input.dim() == 5;

    auto input_batch_norm_shape = is_3d
        ? std::vector<
              int64_t>{1, input_shape[0] * input_shape[1], input_shape[2], 1}
        : std::vector<int64_t>{
              1,
              input_shape[0] * input_shape[1],
              input_shape[2],
              input_shape[3]};
    if (is_5d) {
      input_batch_norm_shape.push_back(input_shape[4]);
    }
    const auto input_reshaped = input.reshape(input_batch_norm_shape);
    const auto weight_reshaped =
        weight.defined() ? weight.repeat(dim0) : at::Tensor();
    const auto gO_reshaped = gO.reshape(input_batch_norm_shape);

    const auto save_mean_reshaped = save_mean.reshape(-1);
    const auto save_invstd_reshaped = save_invstd.reshape(-1);

    const auto ggI_reshaped = grad_in[0].reshape(input_batch_norm_shape);
    const auto ggG_reshaped =
        weight_opt ? grad_in[1].repeat(dim0) : at::Tensor();
    const auto ggB_reshaped = bias_opt ? grad_in[2].repeat(dim0) : at::Tensor();
    const auto [gI, gG, ggP] = batchnorm_double_backward(
        input_reshaped,
        weight_reshaped,
        ggI_reshaped,
        ggG_reshaped,
        ggB_reshaped,
        gO_reshaped,
        std::nullopt, /*running_mean*/
        std::nullopt, /*running_var*/
        true, /*train*/
        eps,
        save_mean_reshaped,
        save_invstd_reshaped,
        mask);

    return {
        gI.reshape(input_shape),
        weight_opt ? gG.index_select(
                         0,
                         torch::arange(
                             dim1, torch::TensorOptions().device(torch::kHPU)))
                   : at::Tensor(),
        ggP.reshape(input_shape),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
  }
};
class InstanceNormAutogradHPU
    : public torch::autograd::Function<InstanceNormAutogradHPU> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const c10::optional<at::Tensor>& weight_opt, // gamma
      const c10::optional<at::Tensor>& bias_opt, // beta
      double eps) {
    auto input_maybe_reshaped = input;
    const auto is_3d = input.dim() == 3;
    if (is_3d) {
      auto new_shape = input.sizes().vec();
      new_shape.push_back(1);
      input_maybe_reshaped = at::reshape(input, new_shape);
    }
    auto [output_maybe_reshaped, mean, istd] = dispatch_instance_norm_hpu(
        input_maybe_reshaped, weight_opt, bias_opt, eps);
    const at::Tensor output = is_3d
        ? at::reshape(output_maybe_reshaped, input.sizes())
        : output_maybe_reshaped;

    ctx->saved_data["weight_opt"] = weight_opt.has_value();
    ctx->saved_data["bias_opt"] = bias_opt.has_value();
    ctx->saved_data["eps"] = eps;
    ctx->save_for_backward(
        {input, mean, istd, weight_opt.value_or(at::Tensor())});
    return output;
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<at::Tensor> grad_in) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto mean = saved[1];
    auto istd = saved[2];
    auto gamma_opt = saved[3];

    const auto eps = ctx->saved_data["eps"].toDouble();
    const auto isWeight = ctx->saved_data["weight_opt"].toBool();
    const auto isBias = ctx->saved_data["bias_opt"].toBool();
    const auto res = InstanceNormBackward::apply(
        input,
        isWeight ? std::make_optional(saved[3]) : std::nullopt,
        grad_in[0],
        mean,
        istd,
        eps,
        isBias);
    return {
        res[0],
        isBias ? res[1] : at::Tensor(),
        isWeight ? res[2] : at::Tensor(),
        at::Tensor()};
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
  PT_OP_INFO(
      " instance_norm:",
      DUMP_9ARGS(
          input,
          weight_opt,
          bias_opt,
          running_mean_opt,
          running_var_opt,
          use_input_stats,
          momentum,
          eps,
          cudnn_enabled));

  return InstanceNormAutogradHPU::apply(input, weight_opt, bias_opt, eps);
}

at::Tensor instance_norm_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool use_input_stats,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  PT_OP_INFO(
      " instance_norm:",
      DUMP_9ARGS(
          input,
          weight_opt,
          bias_opt,
          running_mean_opt,
          running_var_opt,
          use_input_stats,
          momentum,
          eps,
          cudnn_enabled));

  return std::get<0>(
      dispatch_instance_norm_hpu(input, weight_opt, bias_opt, eps));
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("instance_norm", instance_norm_wrap);
}

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  m.impl("instance_norm", instance_norm_autograd_wrap);
}

} // namespace eager
} // namespace habana
