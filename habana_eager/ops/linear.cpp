/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/linear.h"
#include "generated/backend/linear_backward.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "hpu_ops/op_logger.h"
#include "linear.h"

namespace habana {
namespace eager {

at::Tensor linear_fwd_eager_hpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    [[maybe_unused]] const c10::optional<at::Tensor>& bias) {
  PT_EAGER_TRACE;

  auto LinearMeta_ = [](const at::Stack& stack) { return LinearMeta(stack); };

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::linear", {input, weight, bias}};
  hpu_op.SetOutputMetaFn(LinearMeta_);

  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_bwd_eager_hpu(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    std::array<bool, 3> output_mask) {
  PT_EAGER_TRACE;

  auto LinearBackwardMeta_ = [](const at::Stack& stack) {
    return LinearBackwardMeta(stack);
  };
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::linear_backward", {input, grad_output, weight, output_mask}};

  hpu_op.SetOutputMetaFn(LinearBackwardMeta_);
  std::tuple<at::Tensor, at::Tensor, at::Tensor> result = hpu_op.call();
  if (output_mask[2]) {
    return result;
  } else {
    at::Tensor temp = torch::Tensor(); // create a None Tensor for bias_grad
    return std::make_tuple(std::get<0>(result), std::get<1>(result), temp);
  }
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("linear", linear_fwd_eager_hpu);
  m.impl("linear_backward", linear_bwd_eager_hpu);
}
TORCH_LIBRARY_IMPL(aten, HPU, m) { // for cpp test single op call
  m.impl("linear", linear_fwd_eager_hpu);
  m.impl("linear_backward", linear_bwd_eager_hpu);
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def("linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor");
  m.def(
      "linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_linear_backward_hpu(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    std::array<bool, 3> output_mask) {
  PT_EAGER_TRACE;

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::linear_backward", "")
                       .typed<decltype(dispatch_linear_backward_hpu)>();
  return op.call(input, grad_output, weight, output_mask);
}

at::Tensor dispatch_linear_hpu(
    const at::Tensor& input,
    const at::Tensor& other,
    [[maybe_unused]] const c10::optional<at::Tensor>& bias) {
  PT_EAGER_TRACE;

  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::linear", "")
                       .typed<decltype(dispatch_linear_hpu)>();
  return op.call(input, other, bias);
}

class LinearAutogradHPU : public torch::autograd::Function<LinearAutogradHPU> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias) {
    PT_EAGER_TRACE;
    ctx->saved_data["bias"] = bias.has_value() && bias.value().defined();
    ctx->save_for_backward({input, weight});
    return dispatch_linear_hpu(input, weight, bias);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    PT_EAGER_TRACE;
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    bool bias_flag = ctx->saved_data["bias"].toBool();
    std::array<bool, 3> mask = {1, 1, 1};
    if (!bias_flag)
      mask[2] = 0;
    std::tuple<at::Tensor, at::Tensor, at::Tensor> result =
        dispatch_linear_backward_hpu(
            saved_vars[0], grad_output[0], saved_vars[1], mask);
    auto bias_grad = bias_flag ? std::get<2>(result) : torch::Tensor();
    return {std::get<0>(result), std::get<1>(result), bias_grad};
  }
};

at::Tensor linear_autograd_wrap(
    const at::Tensor& input,
    const at::Tensor& other,
    const c10::optional<at::Tensor>& bias) {
  PT_EAGER_TRACE;
  return LinearAutogradHPU::apply(input, other, bias);
}

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  m.impl("linear", linear_autograd_wrap);
}

} // namespace eager
} // namespace habana
