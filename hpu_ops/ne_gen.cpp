/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/ne.h"
#include "hpu_op_helper.h"

namespace habana {
void NE::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  auto outshape = BinaryOutputShape(stack)[0];

  const at::ScalarType& result_type = c10::ScalarType::Bool;

  auto eq = BuildOp(
      graph,
      "equal_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(1)},
      {{outshape, result_type}});

  // not on output of equal
  auto not_equal =
      BuildOp(graph, "not_fwd_i8", {eq[0].get()}, {{outshape, result_type, 0}});

  // output of not is the output of this op
  syn_out(0) = std::move(not_equal[0]);
}
} // namespace habana
