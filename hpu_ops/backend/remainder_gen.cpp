/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "generated/backend/remainder.h"
#include "hpu_ops/div_mod_util.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

void RemainderOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto outshape = BinaryOutputShape(stack)[0];

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
