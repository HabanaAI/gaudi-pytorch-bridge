/**
 * Copyright (c) 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#include "hpu_ops/op_logger.h"

namespace habana {
namespace eager {

at::Tensor linear_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  PT_EAGER_TRACE;
  PT_OP_INFO("linear: ", DUMP_3ARGS(input, weight, bias));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::linear", {input, weight, bias}};
  hpu_op.SetOutputMetaFn(LinearMeta);

  return hpu_op.call();
}

at::Tensor linear_forward_dispatch(
    const at::Tensor& input,
    const at::Tensor& other,
    const c10::optional<at::Tensor>& bias) {
  PT_EAGER_TRACE;
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::linear", "")
                       .typed<decltype(linear_forward_dispatch)>();
  return op.call(input, other, bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_bwd(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    std::array<bool, 3> output_mask) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "linear_bwd: ", DUMP_4ARGS(input, grad_output, weight, output_mask));

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::linear_backward", {input, grad_output, weight, output_mask}};

  hpu_op.SetOutputMetaFn(LinearBackwardMeta);
  std::tuple<at::Tensor, at::Tensor, at::Tensor> result = hpu_op.call();
  if (output_mask[2]) {
    return result;
  } else {
    at::Tensor none_tensor = torch::Tensor();
    return std::make_tuple(
        std::get<0>(result), std::get<1>(result), none_tensor);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_bwd_dispatch(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    std::array<bool, 3> output_mask) {
  PT_EAGER_TRACE;
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::linear_backward", "")
                       .typed<decltype(linear_bwd_dispatch)>();
  return op.call(input, grad_output, weight, output_mask);
}

class LinearFunction : public torch::autograd::Function<LinearFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias) {
    PT_EAGER_TRACE;
    ctx->saved_data["bias"] = bias.has_value() && bias.value().defined();
    ctx->save_for_backward({input, weight});
    return linear_forward_dispatch(input, weight, bias);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    PT_EAGER_TRACE;
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    bool bias_flag = ctx->saved_data["bias"].toBool();
    std::array<bool, 3> mask = {true, true, bias_flag};

    std::tuple<at::Tensor, at::Tensor, at::Tensor> result =
        linear_bwd_dispatch(saved_vars[0], grad_output[0], saved_vars[1], mask);
    auto bias_grad = bias_flag ? std::get<2>(result) : torch::Tensor();
    return {std::get<0>(result), std::get<1>(result), bias_grad};
  }
};

at::Tensor linear_autograd_wrap(
    const at::Tensor& input,
    const at::Tensor& other,
    const c10::optional<at::Tensor>& bias) {
  PT_EAGER_TRACE;
  return LinearFunction::apply(input, other, bias);
}

// When below flag is enabled, aten.linear and aten.matmul decompositions
// are overriden in eager and torch.compile.
static const bool OVERRIDE_LINEAR =
    GET_ENV_FLAG_NEW(PT_HPU_OVERRIDE_LINEAR_MATMUL_EAGER);

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  if (OVERRIDE_LINEAR) {
    m.impl("linear", linear_forward);
    m.impl("linear_backward", linear_bwd);
  }
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def("linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor");
  m.def(
      "linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  if (OVERRIDE_LINEAR) {
    m.impl("linear", linear_forward);
    m.impl("linear_backward", linear_bwd);
  }
}

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  if (OVERRIDE_LINEAR) {
    m.impl("linear", linear_autograd_wrap);
  }
}

} // namespace eager
} // namespace habana
