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
#pragma once

#include <ATen/NamedTensorUtils.h>
#include <ATen/core/TensorBody.h>

namespace habana {
namespace eager {
template <typename Vec>
at::Tensor alias_with_sizes_and_strides(
    const at::Tensor& self,
    const Vec& sizes,
    const Vec& strides) {
  // caller should make sure that sizes and strides are valid for self
  //(storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  TORCH_CHECK(!self.is_quantized());
  at::Tensor self_ = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW,
      at::Storage(self.storage()),
      self.key_set(),
      self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset());
  self_tmp_->set_sizes_and_strides(sizes, strides);

  at::namedinference::propagate_names(self_, self);
  return self_;
}

at::Tensor view_hpu(const at::Tensor& self, c10::SymIntArrayRef size);

// Propagate any view related information
// Tensors passed as copy due to possible access in separate pipeline thread
void view_propagate_permutation(at::Tensor base_t, at::Tensor view_t);

at::Tensor create_base(const at::Tensor& self);

} // namespace eager
} // namespace habana
