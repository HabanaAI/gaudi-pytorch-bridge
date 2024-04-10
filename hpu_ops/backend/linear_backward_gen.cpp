/******************************************************************************
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
  // define output meta
  const auto meta = LinearBackwardMeta(stack);
  size_t size = 0;
  auto params = FillLinearBwdParams(stack, size);
  // define input tensors
  std::vector<synTensor> input_tensor{syn_in(0), syn_in(1), syn_in(2)};
  // define guid name with dtype
  std::string guid =
      get_guid_with_precision("linear_temp_bwd", meta.at(0).dtype);
  // define build op
  std::vector<synapse_helpers::tensor> LinearBwdOP = BuildOp(
      graph,
      guid,
      std::move(input_tensor),
      {{meta.at(0).shape, meta.at(0).dtype, 0},
       {meta.at(1).shape, meta.at(1).dtype, 1},
       {meta.at(2).shape, meta.at(2).dtype, 2}},
      params.get(),
      size);
  // set outputs
  syn_out(0) = std::move(LinearBwdOP[0]);
  syn_out(1) = std::move(LinearBwdOP[1]);
  syn_out(2) = std::move(LinearBwdOP[2]);
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

static const auto& LinearBackwardKernelRegistry = habana::KernelRegistry().add(
    "hpu::linear_backward",
    KERNEL_FN_GLOBAL(habana::LinearBackward));