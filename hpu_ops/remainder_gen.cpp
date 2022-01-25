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

std::shared_ptr<void> FillDivModParams(size_t& size) {
  PARAMS_STUB(ns_DivModKernel::Params);
  // Python div_mod is enabled where remainder returns the same sign of the
  // divisor, except for the zero remainder
  params->isPyCompatible = true;
  return params;
}

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
    auto divMod = BuildOp(
        graph,
        "div_mod_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType()}, {outshape, ScalarType(), 0}},
        params.get(),
        size);
    syn_out(0) = std::move(divMod[1]);
  }
}

} // namespace habana
