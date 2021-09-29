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
void LogAddExp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  // exp on input 0
  auto exp_0 = BuildOp(
      graph,
      "exp_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), false}});

  // exp on input 1
  auto exp_1 = BuildOp(
      graph,
      "exp_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType(), false}});

  // Add on output of exp_0, exp_1
  auto add = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {exp_0[0].get(), exp_1[0].get()},
      {{outshape, ScalarType(), false}});

  // log on output of add
  auto log = BuildOp(
      graph,
      "log_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {add[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  // output of log is the output of this op
  syn_out(0) = std::move(log[0]);
}
} // namespace habana
