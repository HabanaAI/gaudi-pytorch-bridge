/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/where.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {

void WhereBackend::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto shape = ComputeOutputShapes(stack)[0];
  const auto& self = stack_tensor(stack, 1);
  const auto& other = stack_tensor(stack, 2);

  c10::optional<const at::IValue*> output = IsOutputAvailable()
      ? c10::make_optional<const at::IValue*>(&stack.back())
      : c10::nullopt;

  auto dtype_helper =
      habana_helpers::DTypeHelper::binary_op_with_type_promotion(
          {stack.at(1), stack.at(2)}, output, false);

  c10::ScalarType result_type =
      habana_helpers::getInternalDtype(dtype_helper.get_result_dtype());

  std::vector<synapse_helpers::tensor> cast;
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1), syn_in(2)};

  if (habana_helpers::getInternalDtype(self.scalar_type()) != result_type) {
    cast.emplace_back(CastHelper(
        graph, syn_in(1), self.sizes(), self.scalar_type(), result_type));
    inputs[1] = cast[0].get();
  } else if (
      habana_helpers::getInternalDtype(other.scalar_type()) != result_type) {
    cast.emplace_back(CastHelper(
        graph, syn_in(2), other.sizes(), other.scalar_type(), result_type));
    inputs[2] = cast[0].get();
  }

  update_guid_dtype(guid_, result_type);

  auto result = BuildOp(graph, guid_, inputs, {{shape, result_type, 0}});
  syn_out(0) = std::move(result[0]);
}
sizes_vec WhereOutputShape(const at::Stack& stack) {
  at::IntArrayRef cond = stack_tensor(stack, 0).sizes();
  at::IntArrayRef self = stack_tensor(stack, 1).sizes();
  at::IntArrayRef other = stack_tensor(stack, 2).sizes();
  return {at::infer_size(at::infer_size(cond, self), other)};
}

} // namespace habana
