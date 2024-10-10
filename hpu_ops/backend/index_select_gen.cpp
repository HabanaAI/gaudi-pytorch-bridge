/******************************************************************************
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
#include "generated/backend/index_select.h"

constexpr int64_t index_of_self = 0;
constexpr int64_t index_of_dim = 1;
constexpr int64_t index_of_index_position = 2;

namespace habana {

std::shared_ptr<void> FillIndexSelectParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_GatherKernel::Params);

  auto self = stack.at(index_of_self).toTensor();
  auto dim = stack.at(index_of_dim).toInt();
  params->axis = get_dim_in_tpc_order(dim, self.dim());
  return params;
}

OutputMetaDataVector IndexSelectMeta(const at::Stack& stack) {
  auto self = stack.at(index_of_self).toTensor();
  auto dim_ = stack.at(index_of_dim).toInt();
  auto index = stack.at(index_of_index_position).toTensor();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto shape = self.sizes().vec();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  habana_helpers::set_output_hw_scaling_meta(self, meta);
  if (shape.size()) {
    if (self.dim() == index.dim()) {
      meta.shape = index.sizes().vec();
    } else {
      shape.erase(shape.begin() + dim);
      shape.insert(shape.begin() + dim, index.numel());
      meta.shape = shape;
    }
  } else {
    meta.shape = shape;
  }
  return {meta};
}
} // namespace habana
