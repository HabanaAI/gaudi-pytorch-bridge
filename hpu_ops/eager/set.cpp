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

#include "set.h"
#include <ATen/native/Resize.h>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/resize.h"

namespace habana {
namespace eager {

at::Tensor& set_(
    at::Tensor& self,
    at::Storage source,
    at::SymInt storage_offset,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride) {
  at::native::checkSetStorage(self, source, storage_offset, size, stride);

  auto int_storage_offset = storage_offset.as_int_unchecked();
  auto int_stride = asIntArrayRefSlow(stride);
  self.unsafeGetTensorImpl()->set_storage_offset(int_storage_offset);
  at::native::resize_impl_hpu_(
      self.unsafeGetTensorImpl(), asIntArrayRefSlow(size), int_stride);
  return self;
}
} // namespace eager
} // namespace habana

// TODO: These need to be inside habana::eager namespace
// Will be handled by SW-119196
namespace habana_lazy {
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

} // namespace habana_lazy
