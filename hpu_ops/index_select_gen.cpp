/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/index_select.h"
#include "hpu_op_helper.h"

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

sizes_vec IndexSelectOutShape(const at::Stack& stack) {
  auto self = stack.at(index_of_self).toTensor();
  auto dim_ = stack.at(index_of_dim).toInt();
  auto index = stack.at(index_of_index_position).toTensor();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto shape = self.sizes().vec();
  if (shape.size()) {
    if (self.dim() == index.dim()) {
      shape = index.sizes().vec();
    } else {
      shape.erase(shape.begin() + dim);
      shape.insert(shape.begin() + dim, index.numel());
    }
  }
  return {shape};
}
} // namespace habana
