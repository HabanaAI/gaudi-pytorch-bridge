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

#include "habana_eager/ops/as_strided.h"
#include <ATen/native/Resize.h>
#include "habana_eager/ops/view.h"

namespace habana {
namespace eager {
at::Tensor as_strided_hpu(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW,
      c10::Storage(self.storage()),
      self.key_set(),
      self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  habana::eager::view_propagate_permutation(self, result);
  return result;
}
} // namespace eager
} // namespace habana
