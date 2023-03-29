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

#include "copy_from.h"
#include "backend/synapse_helpers/env_flags.h"
#include "eager_op.h"

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

namespace habana {
namespace eager {

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  auto sizes = self.sizes().vec();
  if (self.sizes() != dst.sizes()) {
    dst.resize_(self.sizes());
  }
  return dst.copy_(self);
}

at::Tensor create_base(const at::Tensor& self) {
  auto self_impl = self.unsafeGetTensorImpl();
  auto base_size = (int64_t)(
      habana_helpers::GetNBytes(self_impl) /
      c10::elementSize(habana_helpers::getInternalDtype(self.scalar_type())));
  auto base =
      at::empty(base_size, self.options(), c10::MemoryFormat::Contiguous);
  base.unsafeGetTensorImpl()->set_storage_keep_dtype(self.storage());

  return base;
}

at::Tensor _copy_from_d2h(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  auto self_ = self;
  if (!self.is_contiguous()) {
    auto base = create_base(self);
    habana::eager::EagerOp<at::Tensor> hpu_op{
        "aten::as_strided",
        {base, self.sizes(), self.strides(), self.storage_offset()}};
    self_ = hpu_op.call();
    // restride cpu tensor since synapse will always return contiguous tensor
    dst.unsafeGetTensorImpl()->set_sizes_contiguous(dst.sizes());
    dst.unsafeGetTensorImpl()->set_storage_offset(0);
  }
  if (!GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) &&
      self.scalar_type() == c10::ScalarType::Long) {
    habana_helpers::copy_data_to_host(self_, dst, false);
    unpackData<int32_t, int64_t>(dst);
  } else if (self.scalar_type() == c10::ScalarType::Double) {
    habana_helpers::copy_data_to_host(self_, dst, false);
    unpackData<float, double>(dst);
  } else {
    habana_helpers::copy_data_to_host(self_, dst, non_blocking);
  }
  return dst;
}

at::Tensor add_strided_insert(at::Tensor dst, at::Tensor insert) {
  auto strides = dst.strides().vec();
  auto offset = dst.unsafeGetTensorImpl()->storage_offset();
  auto base = create_base(dst);
  base.unsafeGetTensorImpl()->set_storage_keep_dtype(dst.storage());

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::strided_insert", {base, insert, strides, offset}};
  return hpu_op.call();
}

at::Tensor _copy_from_h2d(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  at::Tensor result;
  // Special handling for Long/Double tensors
  // Downcast sent data (implicitly backend will treat it as Int/Float anyway)
  auto temp_self = self;
  if (!GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) &&
      self.scalar_type() == c10::ScalarType::Long) {
    temp_self = self.to(c10::ScalarType::Int);
  } else if (self.scalar_type() == c10::ScalarType::Double) {
    temp_self = self.to(c10::ScalarType::Float);
  }

  // Handling of CH last case
  // NHWC 'dst' tensor will show up as NCHW but with strides.
  // any consumer node of this tensor will go through JIT IR pass and as_strided
  // node will get added
  temp_self = temp_self.contiguous(self.suggest_memory_format());

  if (!dst.is_contiguous(dst.suggest_memory_format())) {
    // This is done in two steps. First copy the contiguous Host tensor to
    // device. Then invoke strided_insert
    auto insert_t = at::empty(
        temp_self.sizes(), dst.options(), c10::MemoryFormat::Contiguous);
    habana_helpers::copy_data_to_device(temp_self, insert_t, non_blocking);

    result = add_strided_insert(dst, insert_t);
  } else {
    habana_helpers::copy_data_to_device(temp_self, dst, non_blocking);
    result = dst;
  }

  return result;
}

at::Tensor _copy_from_d2d(const at::Tensor& self, const at::Tensor& dst) {
  at::Tensor result;
  if (dst.is_contiguous()) {
    // Since _copy_from is neither inplace nor an out variant but pytorch
    // expects to copy to dst, we treat _copy_from as an out variant in the
    // backend with "dst" as the out tensor
    habana::eager::EagerOp<at::Tensor&> hpu_op{
        "hpu::_copy_from", {self, dst}, {dst.sizes().vec()}, 1};
    result = hpu_op.call(const_cast<at::Tensor&>(dst));
  } else {
    result = add_strided_insert(dst, self);
  }
  return result;
}

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
    result = _copy_from_d2h(self, dst, non_blocking);
  } else if (src_device == at::kCPU) {
    result = _copy_from_h2d(self, dst, non_blocking);
  } else {
    result = _copy_from_d2d(self, dst);
  }
  return result;
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def("_copy_from(Tensor self, Tensor dst) -> Tensor");
  m.def("identity(Tensor self) -> Tensor");
  m.def(
      "strided_insert(Tensor self, Tensor other, int[] stride, int offset) -> (Tensor)");
}
} // namespace eager
} // namespace habana
