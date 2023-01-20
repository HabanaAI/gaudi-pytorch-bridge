/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/isfinite.h"
#include "generated/isinf.h"
#include "generated/isnan.h"

namespace habana {

template <>
IsFiniteInfNan<at::Tensor>::IsFiniteInfNan(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  set_scalar_type(torch::kBool);
}

template <>
at::Tensor IsFiniteInfNan<at::Tensor>::get_result_overrideable() {
  return {};
}

void _IsFiniteInfNan::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto params = FillParams(stack, size);
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{outshape, torch::kBool, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
