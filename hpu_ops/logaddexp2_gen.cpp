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
void LogAddExp2::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  // pow on input 0
  auto pow_out_1 = BuildOp(
      graph,
      "pow2_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), false}});

  // pow on input 1
  auto pow_out_2 = BuildOp(
      graph,
      "pow2_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType(), false}});

  // add on output of add
  auto add = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {pow_out_1[0].get(), pow_out_2[0].get()},
      {{outshape, ScalarType(), false}});

  // log on output of add
  auto log = BuildOp(
      graph,
      "log2_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {add[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  // output of log is the output of this op
  syn_out(0) = std::move(log[0]);
}
} // namespace habana
