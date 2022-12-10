/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/logical_and.h"
#include "generated/logical_not.h"
#include "generated/logical_or.h"
#include "generated/logical_xor.h"

namespace habana {
template <>
LazyLogical<at::Tensor>::LazyLogical(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyLogical<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  auto shape =
      inputs.size() > 1 ? BinaryOutputShape(inputs)[0] : t.sizes().vec();
  return habana_lazy::empty_hpu_lazy(
      shape, t.options().dtype(at::kBool), t.suggest_memory_format(), false);
}

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
