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

template <>
LazyXlogY<at::Tensor>::LazyXlogY(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyXlogY<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(1).toTensor();
  return habana_lazy::empty_hpu_lazy(
      t.sizes(), t.options(), t.suggest_memory_format(), false);
}

void XlogYOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto outshape = XlogYOutputShape(stack)[0];
  if (stack.at(1).isScalar()) {
    auto logy = BuildOp(graph, guid_, {syn_in(1)}, {{1, ScalarType(), false}});

    auto xlogy = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), logy[0].get()},
        {{outshape, ScalarType(), is_output_persistent_list[0], true}});
    syn_out(0) = std::move(xlogy[0]);
  } else {
    auto other_shape = stack_tensor(stack, 1).sizes();
    auto logy = BuildOp(
        graph, guid_, {syn_in(1)}, {{other_shape, ScalarType(), false}});

    auto xlogy = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), logy[0].get()},
        {{outshape, ScalarType(), is_output_persistent_list[0], true}});
    syn_out(0) = std::move(xlogy[0]);
  }
}
} // namespace habana
