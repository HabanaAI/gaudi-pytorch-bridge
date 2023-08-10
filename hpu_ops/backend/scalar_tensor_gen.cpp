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

#include "generated/backend/scalar_tensor.h"

namespace habana {

OutputMetaDataVector ScalarTensorMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.dtype = stack.at(1).isNone() ? at::kFloat : stack.at(1).toScalarType();
  meta.layout = stack.at(2).isNone() ? at::kStrided : stack.at(2).toLayout();

  meta.shape = {}; // This should be a 0 dim tensor

  return {meta};
}

void ScalarTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto scalar = stack.at(0).toScalar();
  auto meta = ScalarTensorMeta(stack)[0];
  auto result = ConstantHelper(graph, scalar, meta.dtype, meta.shape, 0);
  syn_out(0) = std::move(result);
}
} // namespace habana
