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

#include "generated/backend/logical_and.h"
#include "generated/backend/logical_not.h"
#include "generated/backend/logical_or.h"
#include "generated/backend/logical_xor.h"

namespace habana {

static auto CreateLogicalNode(
    OpBackend* op,
    synapse_helpers::graph& graph,
    at::ScalarType compute_type,
    const std::string& guid,
    std::vector<synTensor> inputs,
    at::IntArrayRef outshape,
    at::ScalarType output_dtype,
    synapse_helpers::tensor& syn_out) {
  auto logical_op = OpBackend::BuildNode(
      op, graph, {guid, std::move(inputs), {{outshape, compute_type}}});
  syn_out = OpBackend::BuildCast(
      op, graph, logical_op[0].get(), outshape, compute_type, output_dtype, 0);
}

void LogicalBackend::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (ScalarType() == at::kBool or ScalarType() == at::kChar) {
    return OpBackend::AddNode(graph, stack);
  }

  CreateLogicalNode(
      this,
      graph,
      ScalarType(),
      GetGuid(),
      {syn_in(0), syn_in(1)},
      ComputeOutputShapes(stack)[0],
      GetOutputMetaData(0).dtype,
      syn_out(0));
}

void LogicalNotBackend::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (ScalarType() == at::kBool or ScalarType() == at::kChar) {
    return OpBackend::AddNode(graph, stack);
  }

  CreateLogicalNode(
      this,
      graph,
      ScalarType(),
      GetGuid(),
      {syn_in(0)},
      stack_tensor(stack, 0).sizes(),
      GetOutputMetaData(0).dtype,
      syn_out(0));
}
} // namespace habana
