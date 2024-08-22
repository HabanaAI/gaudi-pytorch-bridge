/*******************************************************************************
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

#include "generated/backend/eye.h"

namespace habana {
OutputMetaDataVector EyeMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const int64_t n = stack.at(0).toInt();
  if (stack.size() == 3) {
    const int64_t m = stack.at(1).toInt();
    meta.dtype = stack_tensor(stack, 2).scalar_type();
    meta.shape = {n, m};
  } else {
    meta.dtype = stack_tensor(stack, 1).scalar_type();
    meta.shape = {n, n};
  }
  return {meta};
}

SharedMetaDataVector EyeSharedMeta(const at::Stack& stack) {
  auto tensor = stack_tensor(stack, stack.size() == 3 ? 2 : 1);
  auto dtype = tensor.scalar_type();

  SharedMetaData matrixDiagSharedMeta("matrix_diagonal_fwd");
  matrixDiagSharedMeta.inputs_data = {{2, dtype}};
  matrixDiagSharedMeta.outputs_data = {{2, dtype}};
  return {matrixDiagSharedMeta};
}

void EyeOpOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  std::vector<synapse_helpers::tensor> eye_out;
  auto meta = EyeMeta(stack)[0];

  auto computeDtype = meta.dtype;
  c10::optional<int> finalResultIndex = 0;
  if (meta.dtype == c10::ScalarType::Long) {
    computeDtype = c10::ScalarType::Int;
    finalResultIndex = c10::nullopt;
  }

  auto constant = ConstantHelper(graph, 1.0f, computeDtype, meta.shape);
  eye_out = BuildOp(
      graph,
      get_guid_with_precision("matrix_diagonal_fwd", computeDtype),
      {constant.get()},
      {{meta.shape, computeDtype, finalResultIndex}});
  if (meta.dtype != computeDtype) {
    auto castNode = BuildCast(
        this, graph, eye_out[0].get(), meta.shape, computeDtype, meta.dtype, 0);
    syn_out(0) = std::move(castNode);
  } else {
    syn_out(0) = std::move(eye_out[0]);
  }
}
} // namespace habana
