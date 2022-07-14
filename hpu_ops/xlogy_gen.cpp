/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/special_xlog1py.h"
#include "generated/xlogy.h"

namespace habana {

sizes_vec XlogYOutputShape(const at::Stack& stack, bool) {
  if (stack.at(1).isScalar()) {
    const torch::Tensor& self = stack_tensor(stack, 0);
    return {self.sizes().vec()};
  } else if (stack.at(0).isScalar()) {
    const torch::Tensor& other = stack_tensor(stack, 1);
    return {other.sizes().vec()};
  }
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& other = stack_tensor(stack, 1);
  return {at::infer_size(self.sizes(), other.sizes())};
}

void XlogYOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = XlogYOutputShape(stack, true)[0];
  auto self_shape = stack_tensor(stack, 0).sizes().vec();
  auto other_shape = stack_tensor(stack, 1).sizes().vec();

  auto self = stack.at(0).toTensor();
  auto other = stack.at(1).toTensor();

  std::vector<synTensor> self_input{syn_in(0)};
  std::vector<synTensor> other_input{syn_in(1)};
  std::unique_ptr<synapse_helpers::tensor> cast;

  // TODO: To remove cast nodes after adding frontend support
  // https://jira.habana-labs.com/browse/SW-86489
  if (self_shape == std::vector<int64_t>({1})) { // Scalar | Tensor
    if (ScalarType() == c10::ScalarType::BFloat16) {
      cast = std::make_unique<synapse_helpers::tensor>(CastHelper(
          graph, syn_in(0), self_shape, self.scalar_type(), ScalarType()));
      self_input = {cast->get()};
    }
  } else if (other_shape == std::vector<int64_t>({1})) { // Tensor | Scalar
    if (ScalarType() == c10::ScalarType::BFloat16) {
      cast = std::make_unique<synapse_helpers::tensor>(CastHelper(
          graph, syn_in(1), other_shape, other.scalar_type(), ScalarType()));
      other_input = {cast->get()};
    }
  }

  auto logy = BuildOp(graph, guid_, other_input, {{other_shape, ScalarType()}});
  auto xlogy = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {self_input[0], logy[0].get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(xlogy[0]);
}
} // namespace habana
