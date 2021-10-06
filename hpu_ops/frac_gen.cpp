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
#include "hpu_op_helper.h"

namespace habana {
void Frac::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  // sign on input 0
  auto sign = BuildOp(
      graph,
      "sign_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), false}});

  // abs on output of sign -> modulus
  auto abs_val = BuildOp(
      graph,
      "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), false}});

  // floor on output of mod
  auto floor_val = BuildOp(
      graph,
      "floor_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {abs_val[0].get()},
      {{outshape, ScalarType(), false}});

  // mul on output of floor & sign
  auto mul = BuildOp(
      graph,
      "mult_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {floor_val[0].get(), sign[0].get()},
      {{outshape, ScalarType(), false}});

  // sub on input & output of mul
  auto sub = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), mul[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  // output of sub is the output of this op
  syn_out(0) = std::move(sub[0]);
}
} // namespace habana
