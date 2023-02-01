/******************************************************************************
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

#include "generated/backend/_foreach_acos.h"
#include "generated/backend/_foreach_add.h"
#include "generated/backend/_foreach_exp.h"
#include "generated/backend/_foreach_zero.h"

namespace habana {

void Foreach::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildOp(
        graph, guid_, {syn_in(i)}, {{tensor.sizes(), tensor.scalar_type(), i}});
    syn_out(i) = std::move(out[0]);
  }
}

void ForeachZero::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out =
        ConstantHelper(graph, 0, tensor.scalar_type(), tensor.sizes(), i);
    syn_out(i) = std::move(out);
  }
}

} // namespace habana
