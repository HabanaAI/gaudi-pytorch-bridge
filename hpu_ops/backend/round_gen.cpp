/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/round.h"

namespace habana {
std::shared_ptr<void> FillRoundParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_RoundKernel::Params);
  static_cast<void>(stack);
  params->roundMode = RoundMode_t::ROUND_HALF_NEAREST_EVEN;
  return params;
}

void RoundDecimal::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  int64_t dec = stack.at(1).toScalar().to<int>();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto dtype = stack_tensor(stack, 0).scalar_type();

  size_t sizes = 0;
  const auto& params = FillRoundParams(stack, sizes);

  if (dec == 0) {
    auto out = BuildOp(
        graph,
        "round_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        {syn_in(0)},
        {{outshape, dtype, 0}},
        params.get(),
        sizes);

    syn_out(0) = std::move(out[0]);

    return;
  }

  std::string guid1 = MULT_GUID;
  std::string guid2 = "div_fwd_";
  if (dec < 0) {
    guid1 = "div_fwd_";
    guid2 = MULT_GUID;
    dec = -dec;
  }

  int const_val = static_cast<int>(pow(10, dec));
  auto constant = ConstantHelper(graph, const_val, dtype, 1);

  auto out1 = BuildOp(
      graph,
      guid1 + habana_helpers::name_suffix_from_type(dtype),
      {syn_in(0), constant.get()},
      {{outshape, dtype}});

  auto out2 = BuildOp(
      graph,
      "round_fwd_" + habana_helpers::name_suffix_from_type(dtype),
      {out1[0].get()},
      {{outshape, dtype}},
      params.get(),
      sizes);

  auto out3 = BuildOp(
      graph,
      guid2 + habana_helpers::name_suffix_from_type(dtype),
      {out2[0].get(), constant.get()},
      {{outshape, dtype, 0}});

  syn_out(0) = std::move(out3[0]);
}
} // namespace habana
