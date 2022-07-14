/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/special_entr.h"
#include "hpu_op_helper.h"

namespace habana {
void SpecialEntr::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  //  Log on input 0
  auto logx = BuildOp(
      graph,
      "log_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType()}});

  // xlog on output of logx
  auto xlogx = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), logx[0].get()},
      {{outshape, ScalarType()}});

  // negative on output of xlogx
  auto nxlogx = BuildOp(
      graph,
      "neg_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {xlogx[0].get()},
      {{outshape, ScalarType()}});

  auto constant = ConstantHelper(graph, 0);
  const at::ScalarType& result_type = c10::ScalarType::Bool;
  // if two tensors have the same size and elements returm true, otherwise False
  auto mask_0 = BuildOp(
      graph,
      "equal_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), constant.get()},
      {{outshape, result_type}});

  // Computes input < other. element-wise and returns a boolean tensor
  auto mask_neg = BuildOp(
      graph,
      "less_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), constant.get()},
      {{outshape, result_type}});

  // Return a tensor of elements selected  x if True  and y if False.
  auto prod_term = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask_0[0].get(), syn_in(0), nxlogx[0].get()},
      {{outshape, ScalarType()}});

  // generating -infinity
  auto ninf = ConstantHelper(graph, -std::numeric_limits<float>::infinity());

  // Return a tensor of elements selected  x if True  and y if False.
  auto output = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask_neg[0].get(), ninf.get(), prod_term[0].get()},
      {{outshape, ScalarType(), 0}});
  syn_out(0) = std::move(output[0]);
}
} // namespace habana
