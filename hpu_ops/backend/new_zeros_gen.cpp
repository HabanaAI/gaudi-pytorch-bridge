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

#include "generated/backend/new_zeros.h"

namespace habana {
OutputMetaDataVector NewZerosMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optionalDtype = stack.at(2).toOptional<at::ScalarType>();
  const at::ScalarType& type = optionalDtype.value_or(self.scalar_type());

  OutputMetaData meta{};
  meta.dtype = type;
  meta.shape = stack.at(1).toIntVector();

  return {meta};
}

SharedMetaDataVector NewZerosSharedMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto optionalDtype = stack.at(2).toOptional<at::ScalarType>();
  auto dtype = optionalDtype.value_or(self.scalar_type());
  auto rank = stack.at(1).toIntVector().size();

  SharedMetaData memsetSharedMeta{"memset"};
  memsetSharedMeta.outputs_data.emplace_back(rank, dtype);
  return {memsetSharedMeta};
}

void NewZerosOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = NewZerosMeta(stack)[0];
  syn_out(0) =
      std::move(BuildOp(graph, "memset", {}, {{meta.shape, meta.dtype, 0}})[0]);
}

} // namespace habana
