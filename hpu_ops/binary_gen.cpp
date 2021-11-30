/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

namespace habana {
sizes_vec BinaryOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  if (stack.at(1).isScalar()) {
    return {self.sizes().vec()};
  }
  const torch::Tensor& other = stack_tensor(stack, 1);
  return {at::infer_size(self.sizes(), other.sizes())};
}

void BinaryOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const at::Tensor& self = stack_tensor(stack, 0);
  const at::Tensor& other = stack_tensor(stack, 1);
  const at::ScalarType& result_type = at::result_type(self, other);

  std::vector<synTensor> binaryop_inputs{syn_in(0), syn_in(1)};
  std::unique_ptr<synapse_helpers::tensor> cast, constant;
  std::vector<synapse_helpers::tensor> mul;

  for (int i = 0; i < 2; i++) {
    const auto& t = stack_tensor(stack, i);
    if (result_type != t.scalar_type()) {
      cast = std::make_unique<synapse_helpers::tensor>(CastHelper(
          graph, syn_in(i), t.sizes(), t.scalar_type(), result_type));
      binaryop_inputs.at(i) = cast->get();
    }
  }

  if (ScalarId().size()) {
    // do alpha mul
    const auto& alpha = ScalarInputs().at(ScalarId()[0]);

    if (ScalarInputs().at(ScalarId()[0]).toFloat() != 1.) {
      constant = std::make_unique<synapse_helpers::tensor>(
          ConstantHelper(graph, alpha, result_type));
      mul = BuildOp(
          graph,
          "mult_fwd_" + habana_helpers::name_suffix_from_type(result_type),
          {syn_in(1), constant->get()},
          {{stack_tensor(stack, 1).sizes(), result_type}});
      binaryop_inputs = {syn_in(0), mul[0].get()};
    }
  }

  auto outshape = BinaryOutputShape(stack)[0];
  // Suffix the promoted type
  guid_ = guid_.substr(0, guid_.find_last_of('_') + 1) +
      habana_helpers::name_suffix_from_type(result_type);

  auto op = BuildOp(
      graph,
      guid_,
      binaryop_inputs,
      {{outshape, result_type, is_output_persistent_list[0], true}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana
