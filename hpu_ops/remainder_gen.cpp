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
#include "generated/remainder.h"
#include "hpu_op_helper.h"

namespace habana {

template <>
RemainderScalarTensor<at::Tensor>::RemainderScalarTensor(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, 1) {}

template <>
at::Tensor RemainderScalarTensor<at::Tensor>::get_result_overrideable() {
  return LazyOp::get_result_overrideable();
}

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
