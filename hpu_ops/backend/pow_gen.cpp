/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/pow.h"

namespace habana {

void PowOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto outshape = self.sizes();
  auto other = stack[1].toScalar();

  if (other.toFloat() == 2.) {
    auto result = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(0)},
        {{outshape, ScalarType(), 0}});

    syn_out(0) = std::move(result[0]);
  } else if (other.toFloat() == 3.) {
    auto temp = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(0)},
        {{outshape, ScalarType()}});
    auto result = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {temp[0].get(), syn_in(0)},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(result[0]);

  } else if (other.toFloat() == 1.) {
    auto result =
        BuildOp(graph, "identity", {syn_in(0)}, {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(result[0]);
  } else {
    auto result = BuildOp(
        graph, guid_, {syn_in(0), syn_in(1)}, {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(result[0]);
  }
}

} // namespace habana
