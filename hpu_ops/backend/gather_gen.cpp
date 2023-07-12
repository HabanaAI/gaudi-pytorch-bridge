/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/gather.h"

namespace habana {

sizes_vec GatherOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto index = stack.at(2).toTensor();
  auto dim_ = stack.at(1).toInt();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  std::vector<int64_t> shape = self.sizes().vec();
  // gather shape check
  auto self_dims = std::max<int64_t>(1, self.dim());
  auto index_dims = std::max<int64_t>(1, index.dim());
  TORCH_CHECK(
      self_dims == index_dims,
      "Index tensor must have the same number of dimensions as input tensor");
  for (int64_t i = 0; i < self_dims; ++i) {
    if (i != dim_) {
      auto index_size = index.dim() == 0 ? 1 : index.size(i);
      auto self_size = self.dim() == 0 ? 1 : self.size(i);
      TORCH_CHECK(
          index_size <= self_size,
          "Size does not match at dimension ",
          i,
          " expected index ",
          index.sizes(),
          " to be smaller than self ",
          self.sizes(),
          " apart from dimension ",
          dim_);
    }
  }
  if (shape.size()) {
    // for gather op, output size is same as index
    if (self.dim() == index.dim()) {
      shape = index.sizes().vec();
    } else {
      // for index_select and other index ops
      shape[dim] = index.numel();
    }
  }
  return {shape};
}

std::shared_ptr<void> FillGatherParams(const at::Stack& stack, size_t& size) {
  auto self = stack.at(0).toTensor();
  int dim_ = stack.at(1).toInt();
  auto dim = get_dim_in_tpc_order(dim_, self.dim());
  at::Tensor indices = stack.at(2).toTensor();
  if (self.dim() != indices.dim()) {
    PARAMS_STUB(ns_GatherKernel::Params);
    params->axis = dim;
    return params;
  }
  PARAMS_STUB(ns_GatherElementsKernel::Params);
  params->axis = dim;
  return params;
}

} // namespace habana
