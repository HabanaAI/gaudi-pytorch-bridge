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
