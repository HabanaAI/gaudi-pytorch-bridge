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

#include "generated/backend/linear_backward.h"
#include "hpu_ops/linear_backward.h"
#include "hpu_ops/op_backend.h"

namespace habana {
OutputMetaDataVector LinearBackwardMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  const auto& weight = stack_tensor(stack, 2);
  const auto& grad_mask = stack.at(3).toBoolList();
  std::vector<int64_t> bias_grad_shape;
  if (grad_mask[2]) {
    bias_grad_shape.push_back(weight.sizes().vec()[0]);
  } else {
    bias_grad_shape.push_back(1);
  }
  OutputMetaData input_meta, weight_meta, bias_meta;

  input_meta.shape = input.sizes().vec();
  input_meta.dtype = input.scalar_type();

  weight_meta.shape = weight.sizes().vec();
  weight_meta.dtype = weight.scalar_type();

  bias_meta.shape = bias_grad_shape;
  bias_meta.dtype = weight.scalar_type();

  return {input_meta, weight_meta, bias_meta};
}

std::shared_ptr<void> FillLinearBwdParams(
    const at::Stack& stack,
    size_t& size) {
  const auto& grad_mask = stack.at(3).toBoolList();
  PARAMS_STUB(ns_LinearBwdKernel::Params);
  params->gradBias = grad_mask[2];

  return params;
}

void LinearBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = LinearBackwardMeta(stack);
  size_t size = 0;
  auto params = FillLinearBwdParams(stack, size);

  std::vector<synTensor> input_tensor{syn_in(0), syn_in(1), syn_in(2)};

  std::string guid =
      get_guid_with_precision("linear_temp_bwd", meta.at(0).dtype);

  std::vector<synapse_helpers::tensor> linear_bwd = BuildOp(
      graph,
      guid,
      std::move(input_tensor),
      {{meta.at(0).shape, meta.at(0).dtype, 0},
       {meta.at(1).shape, meta.at(1).dtype, 1},
       {meta.at(2).shape, meta.at(2).dtype, 2}},
      params.get(),
      size);

  for (size_t i = 0; i < 3; ++i) {
    syn_out(i) = std::move(linear_bwd[i]);
  }
}

LinearBackward::LinearBackward(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "linear_temp_bwd",
          scalar_type,
          {0, 1, 2},
          {},
          {},
          false) {
  SetOutputMetaFn(LinearBackwardMeta);
}
} // namespace habana

// When below flag is enabled, aten.linear and aten.matmul decompositions
// are overriden in eager and torch.compile.
static const auto& LinearBackwardKernelRegistry =
    GET_ENV_FLAG_NEW(PT_HPU_OVERRIDE_LINEAR_MATMUL_EAGER)
    ? habana::KernelRegistry().add(
          "hpu::linear_backward",
          KERNEL_FN_GLOBAL(habana::LinearBackward))
    : habana::KernelRegistry();
