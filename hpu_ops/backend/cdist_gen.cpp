/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/_cdist_forward.h"
#include "hpu_ops/common/batched_matmul_output_shape.h"
#include "hpu_ops/op_backend.h"

namespace habana {
std::shared_ptr<void> FillCdistFwdParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Cdist::Params);
  params->p = stack.at(2).toScalar().toDouble();
  c10::IValue cmVal = stack.at(3);
  params->compute_mode =
      static_cast<CdistComputeMode_t>(cmVal.isInt() ? cmVal.toInt() : 0);
  return params;
}

OutputMetaDataVector CdistFwdMeta(const at::Stack& stack) {
  OutputMetaDataVector metas(1);

  std::array<c10::IntArrayRef, 2> shapes;
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto input = stack_tensor(stack, i);
    if (i == 0) {
      metas[0].dtype = input.scalar_type();
    }

    auto& shape = shapes[i];
    shape = input.sizes();

    TORCH_CHECK(
        shape.size() >= 2,
        "Cdist only supports 2D tensors or above, got: ",
        shape.size(),
        "D");
  }

  metas[0].shape = getBatchMatmulOutShape(shapes[0], shapes[1], false, true);
  return metas;
}

} // namespace habana
