/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "div_mod_util.h"
#include "generated/hpu_op.h"
#include "hpu_op_helper.h"

namespace habana {

void RemainderOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto outshape = BinaryOutputShape(stack, true)[0];

  if (ScalarType() == c10::ScalarType::BFloat16 ||
      (ScalarType() == c10::ScalarType::Float)) {
    // using Reminder kernel
    auto remainder = BuildOp(
        graph,
        "rem_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(remainder[0]);

  } else {
    // Using DivMod kernel
    size_t size = 0;
    const auto& params = FillDivModParams(size);
    auto output = GetDivModOutput(
        this,
        graph,
        syn_in(0),
        syn_in(1),
        /* pyCompatible */ true,
        outshape,
        ScalarType(),
        DIV_MODE_OUTPUT_TYPE::REMAINDER);
    syn_out(0) = std::move(output[1]);
  }
}

} // namespace habana
