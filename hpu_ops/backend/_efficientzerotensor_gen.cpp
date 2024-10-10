/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/_efficientzerotensor.h"
#include "hpu_ops/op_backend.h"

namespace habana {

OutputMetaDataVector EfficientZeroMeta(const at::Stack& stack) {
  auto optionalDtype = stack.at(1).toOptional<at::ScalarType>();
  const at::ScalarType& type =
      optionalDtype.value_or(torch::get_default_dtype_as_scalartype());
  OutputMetaData meta{};

  meta.dtype = type;
  meta.shape = stack.at(0).toIntVector();

  return {meta};
}

void EfficientZeroTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = EfficientZeroMeta(stack)[0];
  syn_out(0) =
      std::move(BuildOp(graph, "memset", {}, {{meta.shape, meta.dtype, 0}})[0]);
}

} // namespace habana
