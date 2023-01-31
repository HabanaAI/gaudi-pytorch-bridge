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

#include "generated/cumprod.h"
#include "generated/cumsum.h"

namespace habana {

template <>
LazyCumsum<at::Tensor>::LazyCumsum(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyCumsum<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  const auto& options = inputs.at(2).isNone()
      ? isIntegralType(t.scalar_type(), true)
          ? t.options().dtype(c10::ScalarType::Long)
          : t.options()
      : t.options().dtype(inputs.at(2).toScalarType());
  return habana_lazy::empty_hpu_lazy(
      t.sizes(), options, t.suggest_memory_format(), false);
}

std::shared_ptr<void> FillCumsumParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_CumSumKernel::Params);
  auto self = stack.at(0).toTensor();
  auto dim = at::maybe_wrap_dim(stack.at(1).toInt(), self.dim(), true);
  params->axis = static_cast<int>(self.sizes().vec().size() - dim - 1);

  return params;
}

void CumsumHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  at::ScalarType dtype =
      stack.at(2).isNone() ? ScalarType() : stack.at(2).toScalarType();

  if (habana_helpers::is_downcast_to_int_needed(dtype)) {
    dtype = at::ScalarType::Int;
  }

  if (dtype == at::ScalarType::Double) {
    dtype = at::ScalarType::Float;
  }

  if (dtype == ScalarType() || ScalarType() == at::ScalarType::Double) {
    return OpBackend::AddNode(graph, stack);
  }

  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto cast = CastHelper(graph, syn_in(0), outshape, ScalarType(), dtype);

  size_t size = 0;
  const auto& params = FillCumsumParams(stack, size);
  const std::string& guid = guid_.substr(0, guid_.find_last_of('_') + 1);
  const std::string& cast_to = habana_helpers::name_suffix_from_type(dtype);

  auto op = BuildOp(
      graph,
      guid + cast_to,
      {cast.get()},
      {{outshape, dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(op.at(0));
}

} // namespace habana
