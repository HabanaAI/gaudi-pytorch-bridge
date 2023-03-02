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

#include "backend/synapse_helpers/env_flags.h"
#include "eager_op.h"
#include "habana_kernels/lazy_kernels_declarations.h"

namespace {
// Backend it treating Long(int64) as Int(int32) and Double as Float.
// In Lazy, we track this under the hood, and implicitly up/down cast data.
// In Eager, we cannot track it, so we 'unpack' tensors received from HPU
// with simple reinterpret_cast trick.
template <class From, class To>
void unpackData(const at::Tensor& t) {
  static_assert(sizeof(From) <= sizeof(To));
  From* from_ptr = reinterpret_cast<From*>(t.data_ptr());
  To* to_ptr = reinterpret_cast<To*>(t.data_ptr());
  for (int idx = t.numel() - 1; idx >= 0; --idx) {
    to_ptr[idx] = from_ptr[idx];
  }
}
} // namespace

// TODO Put under namesapce habana::eager
namespace habana_lazy {
at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  const auto src_device = self.device().type();
  const auto dst_device = dst.device().type();
  at::Tensor result;
  // Special handling for Long/Double tensors
  // Unpack received data (implicitly received as Int/Float half of buffer)
  if (dst_device == at::kCPU) {
    if (!GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) &&
        self.scalar_type() == c10::ScalarType::Long) {
      habana_helpers::copy_data_to_host(self, dst, false);
      unpackData<int32_t, int64_t>(dst);
    } else if (self.scalar_type() == c10::ScalarType::Double) {
      habana_helpers::copy_data_to_host(self, dst, false);
      unpackData<float, double>(dst);
    } else {
      habana_helpers::copy_data_to_host(self, dst, non_blocking);
    }
    result = dst;
  } else if (src_device == at::kCPU) {
    // Special handling for Long/Double tensors
    // Downcast sent data (implicitly backend will treat it as Int/Float anyway)
    if (!GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) &&
        self.scalar_type() == c10::ScalarType::Long) {
      auto temp_self = self.to(c10::ScalarType::Int);
      habana_helpers::copy_data_to_device(temp_self, dst, non_blocking);
    } else if (self.scalar_type() == c10::ScalarType::Double) {
      auto temp_self = self.to(c10::ScalarType::Float);
      habana_helpers::copy_data_to_device(temp_self, dst, non_blocking);
    } else {
      habana_helpers::copy_data_to_device(self, dst, non_blocking);
    }
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
