/**
* Copyright (c) 2021-2024 Intel Corporation
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
#include "generated/backend/threshold.h"
#include "generated/backend/threshold_backward.h"

namespace habana {

std::shared_ptr<void> FillThresholdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_ReluKernel::ParamsV2);
  params->threshold.f = stack.at(1).toScalar().to<float>();
  params->replacementValue.f = stack.at(2).toScalar().to<float>();
  return params;
}

std::shared_ptr<void> FillThresholdBwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_ReluKernel::Params);
  params->threshold.f =
      stack.at(2).isNone() ? 0 : stack.at(2).toScalar().to<float>();
  return params;
}

OutputMetaDataVector ThresholdBwdMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

void ThresholdBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 3,
      "Incorrect size of inputs expected for threshold operator");

  TORCH_CHECK(stack[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(stack[1].isTensor(), "Input arg2 type expected to be tensor");

  auto grad_output = stack[0].toTensor();
  auto self = stack[1].toTensor();
  TORCH_CHECK(grad_output.sizes() == self.sizes(), "Input sizes must be equal");

  return OpBackend::AddNode(graph, stack);
}

} // namespace habana
