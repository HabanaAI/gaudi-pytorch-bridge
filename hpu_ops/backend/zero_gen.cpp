/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/zero.h"

namespace habana {

SharedMetaDataVector ZeroSharedMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dtype = self.scalar_type();
  auto rank = self.dim();

  SharedMetaData memsetSharedMeta{"constant"};
  memsetSharedMeta.outputs_data.emplace_back(rank, dtype);
  return {memsetSharedMeta};
}

void ZeroHpuLazyOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto outshape = self.sizes();
  const auto type = self.scalar_type();
  const auto value = 0;
  const auto final_output_index = 0;
  syn_out(0) = ConstantHelper(graph, value, type, outshape, final_output_index);
}

} // namespace habana
