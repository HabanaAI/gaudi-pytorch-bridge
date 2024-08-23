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
#include "habana_eager/ops/masked_select.h"
#include <c10/core/SymIntArrayRef.h>
#include "habana_kernels/resize.h"
#include "hpu_ops/op_logger.h"
namespace habana {
namespace eager {

at::Tensor masked_select_eager(const at::Tensor& self, const at::Tensor& mask) {
  TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool,
              "masked_select: expected BoolTensor for mask");
  auto new_size = at::infer_size(self.sizes(), mask.sizes());
  auto new_mask = at::broadcast_to(mask, new_size);
  auto new_self = at::broadcast_to(self, new_size);

  return at::index(new_self, {new_mask});
}

at::Tensor& masked_select_out_eager(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& out) {
  auto output = masked_select_eager(self, mask);
  std::vector<int64_t> out_shape{output.sizes().vec()[0]};
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = out.unsafeGetTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(
        c10::IntArrayRef(out_shape));
  }
  out.copy_(output);
  return out;
}

} // namespace eager
} // namespace habana