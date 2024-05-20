/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/_foreach_log10.h"
#include "generated/backend/log10.h"

namespace habana {

void ForeachLog10::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const at::ScalarType dtype = isIntegralType(tensor.scalar_type(), true)
        ? torch::kFloat32
        : tensor.scalar_type();
    auto out = BuildOp(
        graph,
        get_guid_with_precision("log10_fwd", dtype),
        {syn_in(i)},
        {{{tensor.sizes()}, dtype, i}});
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
