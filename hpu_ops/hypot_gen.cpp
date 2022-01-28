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
void Hypot::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape_self = stack_tensor(stack, 0).sizes();
  const auto& outshape_other = stack_tensor(stack, 1).sizes();
  auto result_outshape = BinaryOutputShape(stack, true)[0];

  // mul on input 0
  auto mul_1 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(0)},
      {{outshape_self, ScalarType()}});

  // mul on input 1
  auto mul_2 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1), syn_in(1)},
      {{outshape_other, ScalarType()}});

  // addsquare on output of mul
  auto add = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mul_1[0].get(), mul_2[0].get()},
      {{result_outshape, ScalarType()}});

  // sqrt on output of addsquare
  auto sqrt = BuildOp(
      graph,
      "sqrt_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {add[0].get()},
      {{result_outshape, ScalarType(), is_output_persistent_list[0], true}});

  // output of sqrt is the output of this op
  syn_out(0) = std::move(sqrt[0]);
}
} // namespace habana
