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
void LogCumsumExp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  // exp on input 0
  auto exp = BuildOp(
      graph,
      "exp_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType()}});

  // Fill params for cumsum
  size_t size = 0;
  const auto& cumsum_params = FillCumsumParams(stack, size);

  // cumsum on output of exp
  auto cumsum = BuildOp(
      graph,
      "cumsum_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {exp[0].get()},
      {{outshape, ScalarType()}},
      cumsum_params.get(),
      size);

  // log on output of cumsum
  auto log = BuildOp(
      graph,
      "log_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {cumsum[0].get()},
      {{outshape, ScalarType(), 0}});

  // output of log is the output of this op
  syn_out(0) = std::move(log[0]);
}
} // namespace habana
