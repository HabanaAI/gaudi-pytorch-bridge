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

#include "habana_kernels/lazy_kernels_declarations.h"

// TODO Put under namesapce habana::eager
namespace habana_lazy {
at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  const auto src_device = self.device().type();
  const auto dst_device = dst.device().type();
  HABANA_ASSERT(src_device != dst_device);
  if (dst_device == at::kCPU) {
    habana_helpers::copy_data_to_host(self, dst, non_blocking);
  } else {
    habana_helpers::copy_data_to_device(self, dst, non_blocking);
  }
  return dst;
}
} // namespace habana_lazy
