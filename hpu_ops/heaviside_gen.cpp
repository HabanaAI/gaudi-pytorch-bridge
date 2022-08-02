/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/heaviside.h"
#include "hpu_op_helper.h"

namespace habana {

std::shared_ptr<void> FillConstantParams(
    const at::Stack& stack,
    size_t& size,
    int value) {
  PARAMS_STUB(ns_ConstantKernel::Params);

  if (stack[0].toTensor().scalar_type() == c10::ScalarType::Int) {
    get<int>(params->constant) = value;
  } else {
    get<float>(params->constant) = value;
  }
  return params;
}

void Heaviside::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& comparision_outshape = stack_tensor(stack, 0).sizes();
  const auto final_outshape = BinaryOutputShape(stack)[0];
  size_t size = 0;
  const auto& params = FillConstantParams(stack, size, 1);

  auto const_one = BuildOp(
      graph,
      "constant_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {},
      {{1, ScalarType()}},
      params.get(),
      size);

  size = 0;
  const auto& params1 = FillConstantParams(stack, size, 0);

  auto const_zero = BuildOp(
      graph,
      "constant_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {},
      {{1, ScalarType()}},
      params1.get(),
      size);

  const at::ScalarType& result_type = c10::ScalarType::Bool;

  auto less_than_zero = BuildOp(
      graph,
      "less_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), const_zero[0].get()},
      {{comparision_outshape, result_type}});

  auto greater_than_zero = BuildOp(
      graph,
      "greater_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), const_zero[0].get()},
      {{comparision_outshape, result_type}});

  auto where_inner = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {greater_than_zero[0].get(), const_one[0].get(), syn_in(1)},
      {{final_outshape, ScalarType()}});

  auto where_outer = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {less_than_zero[0].get(), const_zero[0].get(), where_inner[0].get()},
      {{final_outshape, ScalarType(), 0}});

  // output of where_outer is the output of this op
  syn_out(0) = std::move(where_outer[0]);
}
} // namespace habana
