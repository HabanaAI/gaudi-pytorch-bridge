/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
