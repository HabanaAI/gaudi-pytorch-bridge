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
#include "hpu_ops/matmul.h"
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include "backend/helpers/tensor_utils.h"
#include "common/dump_args.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/op_logger.h"

namespace habana {
namespace eager {

at::Tensor matmul_forward(const at::Tensor& self, const at::Tensor& other) {
  PT_EAGER_TRACE;
  PT_OP_INFO("matmul: ", DUMP_2ARGS(self, other));
  habana::eager::EagerOp<at::Tensor> hpu_op{"hpu::matmul", {self, other}};
  hpu_op.SetOutputMetaFn(MatmulMeta);
  return hpu_op.call();
}

at::Tensor matmul_forward_dispatch(
    const at::Tensor& self,
    const at::Tensor& other) {
  PT_EAGER_TRACE;
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::matmul", "")
                       .typed<decltype(matmul_forward_dispatch)>();

  return op.call(self, other);
}

std::tuple<at::Tensor, at::Tensor> matmul_bwd(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& other) {
  PT_EAGER_TRACE;
  PT_OP_INFO("matmul_bwd: ", DUMP_3ARGS(grad_output, self, other));
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op(
      "hpu::matmul_bwd",
      {grad_output, self, other},
      {self.sizes().vec(), other.sizes().vec()});
  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor> matmul_bwd_dispatch(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& other) {
  PT_EAGER_TRACE;
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::matmul_bwd", "")
                       .typed<decltype(matmul_bwd_dispatch)>();
  return op.call(grad_output, self, other);
}

struct MatmulFunction : public torch::autograd::Function<MatmulFunction> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& self,
      const at::Tensor& other) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto output = matmul_forward_dispatch(self, other);
    ctx->save_for_backward({self, other});
    return output;
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();

    auto grads =
        matmul_bwd_dispatch(grad_output[0], saved_vars[0], saved_vars[1]);
    return {std::get<0>(grads), std::get<1>(grads)};
  }
};

at::Tensor matmul_autograd_wrap(
    const at::Tensor& self,
    const at::Tensor& other) {
  PT_EAGER_TRACE;
  return MatmulFunction::apply(self, other);
}

// When below flag is enabled, aten.linear and aten.matmul decompositions
// are overriden in eager and torch.compile.
static const bool OVERRIDE_LINEAR =
    GET_ENV_FLAG_NEW(PT_HPU_OVERRIDE_LINEAR_MATMUL_EAGER);

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  if (OVERRIDE_LINEAR) {
    m.impl("hpu::matmul", matmul_forward);
    m.impl("hpu::matmul_bwd", matmul_bwd);
  }
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def("hpu::matmul(Tensor self, Tensor other) -> Tensor");
  m.def(
      "hpu::matmul_bwd(Tensor grad_out, Tensor self, Tensor other) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  if (OVERRIDE_LINEAR) {
    m.impl("matmul", matmul_forward);
  }
}

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  if (OVERRIDE_LINEAR) {
    m.impl("matmul", matmul_autograd_wrap);
  }
}

} // namespace eager
} // namespace habana
