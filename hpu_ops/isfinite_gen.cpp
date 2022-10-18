/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/isfinite.h"

namespace habana {

template <>
IsFinite<at::Tensor>::IsFinite(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  set_scalar_type(torch::kBool);
}

template <>
at::Tensor IsFinite<at::Tensor>::get_result_overrideable() {
  return {};
}

void _IsFinite::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto result =
      BuildOp(graph, guid_, {syn_in(0)}, {{outshape, torch::kBool, 0}});
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
