/*******************************************************************************
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

#include "eager_op.h"
#include "habana_kernels/lazy_kernels_declarations.h"

// TODO Put under namesapce habana::eager
namespace habana_lazy {
at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  const auto src_device = self.device().type();
  const auto dst_device = dst.device().type();
  at::Tensor result;
  if (dst_device == at::kCPU) {
    habana_helpers::copy_data_to_host(self, dst, non_blocking);
    result = dst;
  } else if (src_device == at::kCPU) {
    habana_helpers::copy_data_to_device(self, dst, non_blocking);
    result = dst;
  } else {
    // Since _copy_from is neither inplace nor an out variant but pytorch
    // expects to copy to dst, we treat _copy_from as an out variant in the
    // backend with "dst" as the out tensor
    habana::eager::EagerOp<at::Tensor&> hpu_op{
        "hpu::_copy_from", {self, dst}, {dst.sizes().vec()}, 1};
    result = hpu_op.call(const_cast<at::Tensor&>(dst));
  }
  return result;
}
TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def("_copy_from(Tensor self, Tensor dst) -> Tensor");
}
} // namespace habana_lazy
