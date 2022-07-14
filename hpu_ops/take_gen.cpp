/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/take.h"
#include "hpu_op_helper.h"

namespace habana {

void TakeOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  ns_GatherKernel::Params params{};

  // (M, N) -> (MN)
  auto reshape_outshape = self.numel();
  auto reshape =
      ReshapeHelper(graph, syn_in(0), reshape_outshape, ScalarType());

  // Gathers values along an axis
  auto gatherkernel = BuildOp(
      graph,
      "gather_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {reshape.get(), syn_in(1)},
      {{outshape, ScalarType(), 0}},
      &params,
      sizeof(params));

  syn_out(0) = std::move(gatherkernel[0]);
}
} // namespace habana
