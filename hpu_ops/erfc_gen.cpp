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

void Erfc::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto erf = BuildOp(
      graph,
      "erf_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), false}});

  auto constant = ConstantHelper(graph, 1, ScalarType(), outshape);

  auto erfc = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {constant.get(), erf[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  syn_out(0) = std::move(erfc[0]);
}
} // namespace habana
