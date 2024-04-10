/*
******************************************************************************
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
  // define output meta
  const auto meta = LinearMeta(stack)[0];
  // define input tensors
  std::vector<synTensor> input_tensor{syn_in(0), syn_in(1)};
  // based on bias(True or False) we will push to input_tensors
  if (stack.at(2).isTensor()) {
    input_tensor.push_back(syn_in(2));
  }
  // define guid name with dtype
  std::string guid = get_guid_with_precision("linear_fwd", meta.dtype);
  // define build op
  std::vector<synapse_helpers::tensor> LinearOP = BuildOp(
      graph, guid, std::move(input_tensor), {{meta.shape, meta.dtype, 0}});
  // set the output
  syn_out(0) = std::move(LinearOP[0]);
}

Linear::Linear(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "linear_fwd", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(LinearMeta);
}
} // namespace habana

static const auto& LinearKernelRegistry = habana::KernelRegistry().add(
    "hpu::linear",
    KERNEL_FN_GLOBAL(habana::Linear));