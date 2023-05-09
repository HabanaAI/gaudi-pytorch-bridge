/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
    auto out = BuildOp(
        graph,
        "log10_fwd_" +
            habana_helpers::name_suffix_from_type(tensor.scalar_type()),
        {syn_in(i)},
        {{{tensor.sizes()}, tensor.scalar_type(), i}});
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
