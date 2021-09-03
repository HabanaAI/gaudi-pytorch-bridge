/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <utility>

#include "generated/hpu_op.h"

namespace habana {

template <>
LazyCumsum<at::Tensor>::LazyCumsum(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyCumsum<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  const auto& options = inputs.at(2).isNone()
      ? t.options()
      : t.options().dtype(inputs.at(2).toScalarType());
  return habana_lazy::empty_hpu_lazy(
      t.sizes(), options, t.suggest_memory_format(), false);
}

std::shared_ptr<void> HabanaOperatorHelper::FillCumsumParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_CumSumKernel::Params);
  auto self = stack.at(0).toTensor();
  auto dim = at::maybe_wrap_dim(stack.at(1).toInt(), self.dim(), true);
  params->axis = static_cast<int>(self.sizes().vec().size() - dim - 1);

  return params;
}

void CumsumHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const at::ScalarType& dtype =
      stack.at(2).isNone() ? ScalarType() : stack.at(2).toScalarType();

  if (dtype == ScalarType()) {
    return HabanaOperatorHelper::AddNode(
        graph, stack, is_output_persistent_list);
  }

  const auto& outshape = stack_tensor(stack, 0).sizes();
  const std::string& cast_from =
      habana_helpers::name_suffix_from_type(ScalarType());
  const std::string& cast_to = habana_helpers::name_suffix_from_type(dtype);
  auto cast = BuildOp(
      graph,
      "cast_" + cast_from + "_to_" + cast_to,
      {syn_in(0)},
      {{outshape, dtype, false}});

  size_t size = 0;
  const auto& params = FillCumsumParams(stack, size);
  const std::string& guid = guid_.substr(0, guid_.find_last_of('_') + 1);
  auto op = BuildOp(
      graph,
      guid + cast_to,
      {cast.at(0).get()},
      {{outshape, dtype, is_output_persistent_list[0], true}},
      params.get(),
      size);
  syn_out(0) = std::move(op.at(0));
}

} // namespace habana
