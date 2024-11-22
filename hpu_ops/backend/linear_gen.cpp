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
#include "generated/backend/linear.h"
#include "hpu_ops/linear.h"
#include "hpu_ops/op_backend.h"

namespace habana {

OutputMetaDataVector LinearMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  const auto& weight = stack.at(1).toTensor();
  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = input.sizes().vec();
  meta.shape[input.dim() - 1] = weight.sizes().vec()[0];
  // Condition check to detect input with incompatible shapes
  // Number of dimensions in matrix 1 can vary
  int mat1_dim0 = 1, dim_i = 0;
  for (; dim_i < input.dim() - 1; ++dim_i)
      mat1_dim0 *= input.sizes().vec()[dim_i];
  TORCH_CHECK(
      input.sizes().vec()[input.dim()-1] == weight.sizes().vec()[1], "matrix 1 and matrix 2 shapes cannot be multiplied (",
      mat1_dim0, "x", input.sizes().vec()[input.dim()-1], " and ",
      weight.sizes().vec()[1], "x", weight.sizes().vec()[0], ")");

  return {meta};
}

void Linear::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto meta = LinearMeta(stack)[0];
  std::vector<synTensor> input_tensor{syn_in(0), syn_in(1)};

  if (stack.at(2).isTensor()) {
    input_tensor.push_back(syn_in(2));
  }

  std::string guid = get_guid_with_precision("linear_fwd", meta.dtype);

  std::vector<synapse_helpers::tensor> linear = BuildOp(
      graph, guid, std::move(input_tensor), {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(linear[0]);
}

Linear::Linear(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "linear_fwd", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(LinearMeta);
}
} // namespace habana

// When below flag is enabled, aten.linear and aten.matmul decompositions
// are overriden in eager and torch.compile.
static const auto& LinearKernelRegistry =
    GET_ENV_FLAG_NEW(PT_HPU_OVERRIDE_LINEAR_MATMUL_EAGER)
    ? habana::KernelRegistry().add(
          "hpu::linear",
          KERNEL_FN_GLOBAL(habana::Linear))
    : habana::KernelRegistry();
