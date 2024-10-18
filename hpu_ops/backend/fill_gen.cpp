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
SharedMetaDataVector FillScalarSharedMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  const auto dtype = input.scalar_type();
  const auto rank = input.dim();
  SharedMetaTensor inOutTensor{rank, dtype};
  if (rank > 1) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data = {inOutTensor};
    return {constantSharedMeta};
  } else {
    SharedMetaData memcpySharedMeta{"memcpy"};
    memcpySharedMeta.inputs_data = {inOutTensor};
    memcpySharedMeta.outputs_data = {inOutTensor};
    return {memcpySharedMeta};
  }
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
