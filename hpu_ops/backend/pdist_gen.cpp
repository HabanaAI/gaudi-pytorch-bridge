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

#include "generated/backend/_pdist_forward.h"
#include "hpu_ops/op_backend.h"

namespace habana {
std::shared_ptr<void> FillPdistFwdParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Pdist::Params);
  params->p = stack.at(1).toScalar().toDouble();
  return params;
}

OutputMetaDataVector PdistFwdMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shape = self.sizes().vec();

  TORCH_CHECK(
      shape.size() == 2,
      "pdist only supports 2D tensors, got: ",
      shape.size(),
      "D");

  auto d = shape[0];
  OutputMetaDataVector metas(1);
  metas[0].shape = {(d >= 2) ? d * (d - 1) / 2 : 0};
  metas[0].dtype = self.scalar_type();

  return metas;
}

} // namespace habana
