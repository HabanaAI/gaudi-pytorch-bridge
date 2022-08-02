/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/signbit.h"
#include "hpu_op_helper.h"

namespace habana {

template <>
LazySetDtypeBool<at::Tensor>::LazySetDtypeBool(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  set_scalar_type(c10::ScalarType::Bool);
}

template <>
at::Tensor LazySetDtypeBool<at::Tensor>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return {};
}

void SignBit::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  const at::ScalarType& result_type = stack_tensor(stack, 0).scalar_type();

  size_t size = 0;
  PARAMS_STUB(ns_ConstantKernel::Params);
  get<int>(params->constant) = 0;
  const at::ScalarType& result_type2 = c10::ScalarType::Bool;

  auto const_zero = BuildOp(
      graph,
      "constant_" + habana_helpers::name_suffix_from_type(result_type),
      {},
      {{1, result_type}},
      params.get(),
      size);

  // lesser than on input 0 & zero
  auto output = BuildOp(
      graph,
      "less_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), const_zero[0].get()},
      {{outshape, result_type2, 0}});

  // output of log is the output of this op
  syn_out(0) = std::move(output[0]);
}
} // namespace habana
