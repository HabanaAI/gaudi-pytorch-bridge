/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/fill.h"

namespace habana {
SharedMetaDataVector FillSharedMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  const auto& other = stack_tensor(stack, 1);
  auto dtype = input.scalar_type();

  SharedMetaData meta{"broadcast"};
  meta.inputs_data = {{other.dim(), dtype}};
  meta.outputs_data = {{input.dim(), dtype}};

  return {meta};
}

SharedMetaDataVector FillScalarSharedMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();

  SharedMetaData meta{"memcpy"};
  SharedMetaTensor inOutTensor{rank, dtype};
  meta.inputs_data = {inOutTensor};
  meta.outputs_data = {inOutTensor};

  return {meta};
}

void Fill::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto broadcast =
      BuildOp(graph, "broadcast", {syn_in(1)}, {{outshape, ScalarType(), 0}});

  // output of broadcast is the output of this op
  syn_out(0) = std::move(broadcast[0]);
}

void FillScalar::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto other = stack.at(1).toScalar();

  // If self is a ZST then return it as it is since there is nothing to fill
  if (!self.numel()) {
    const auto& outshape = stack_tensor(stack, 0).sizes();
    auto copy =
        BuildOp(graph, "memcpy", {syn_in(0)}, {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(copy[0]);
  } else {
    const auto& outshape = self.sizes();
    auto result = ConstantHelper(graph, other, ScalarType(), outshape, 0);
    syn_out(0) = std::move(result);
  }
}
} // namespace habana
