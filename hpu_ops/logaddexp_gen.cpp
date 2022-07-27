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
void LogAddExp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape_self = stack_tensor(stack, 0).sizes();
  const auto& outshape_other = stack_tensor(stack, 1).sizes();
  auto result_outshape = ComputeOutputShapes(stack, true)[0];

  // exp on input 0
  auto exp_0 = BuildOp(
      graph,
      "exp_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape_self, ScalarType()}});

  // exp on input 1
  auto exp_1 = BuildOp(
      graph,
      "exp_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1)},
      {{outshape_other, ScalarType()}});

  // Add on output of exp_0, exp_1
  auto add = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {exp_0[0].get(), exp_1[0].get()},
      {{result_outshape, ScalarType()}});

  // log on output of add
  auto log = BuildOp(
      graph,
      "log_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {add[0].get()},
      {{result_outshape, ScalarType(), 0}});

  // output of log is the output of this op
  syn_out(0) = std::move(log[0]);
}
} // namespace habana
