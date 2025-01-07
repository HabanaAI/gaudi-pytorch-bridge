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
#include "habana_eager/ops/set.h"
#include "habana_kernels/resize.h"

#include <ATen/native/Resize.h>
#include <c10_ver/core/SymIntArrayRef.h>

namespace habana {
namespace eager {

at::Tensor& set_source_Storage_storage_offset(
    at::Tensor& self,
    at::Storage source,
    at::SymInt storage_offset,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride) {
  at::native::checkSetStorage(self, source, storage_offset, size, stride);

  auto int_storage_offset = storage_offset.as_int_unchecked();
  c10::optional<at::IntArrayRef> stride_opt = stride.data() != nullptr
      ? c10::optional<at::IntArrayRef>(C10_AS_INTARRAYREF_SLOW(stride))
      : c10::nullopt;

  auto hb_tmeta{habana::get_tensor_extra_meta(self)};
  hb_tmeta->set_tensor_pipelined();

  self.unsafeGetTensorImpl()->set_storage_offset(int_storage_offset);
  at::native::resize_impl_hpu_(
      self.unsafeGetTensorImpl(), C10_AS_INTARRAYREF_SLOW(size), stride_opt);
  return self;
}

at::Tensor& set_source_Storage(at::Tensor& self, at::Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  return self.set_(source, 0, new_size, {});
}

at::Tensor& set_source_Tensor(at::Tensor& self, const at::Tensor& source) {
  if (self.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    return self.set_(
        source.storage(),
        source.storage_offset(),
        source.sizes(),
        source.strides());
  }
  return self;
}

at::Tensor& set_(at::Tensor& self) {
  caffe2::TypeMeta dtype = self.dtype();
  at::Storage storage(
      at::Storage::use_byte_size_t(), 0, c10::GetAllocator(at::kHPU), true);
  self.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == self.dtype());
  return self;
}

} // namespace eager
} // namespace habana
