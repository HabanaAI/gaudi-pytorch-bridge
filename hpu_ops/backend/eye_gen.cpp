/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

void EyeOpOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  std::vector<synapse_helpers::tensor> eye_out;
  auto meta = EyeMeta(stack)[0];

  auto constant = ConstantHelper(graph, 1.0f, meta.dtype, meta.shape);

  eye_out = BuildOp(
      graph,
      get_guid_with_precision("matrix_diagonal_fwd", meta.dtype),
      {constant.get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(eye_out[0]);
}
} // namespace habana
