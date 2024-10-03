/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include <ATen/CPUFunctions.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Optional.h>
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "hpu_ops/op_logger.h"

namespace habana {
namespace eager {

static inline at::Device ensure_has_index(at::Device device) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

at::Tensor pin_memory_hpu(const at::Tensor& self, at::Device device) {
  ensure_has_index(device);
  auto* allocator = habana::getPinnedMemoryAllocator();
  auto storage = at::Storage(
      at::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

bool is_pinned_hpu(const at::Tensor& self, at::Device device) {
  ensure_has_index(device);
  return habana::PinnedMemoryAllocator_is_pinned(self.data_ptr());
}

} // namespace eager
} // namespace habana

namespace hpu_wrap {
at::Tensor _pin_memory(
    const at::Tensor& self,
    ::std::optional<at::Device> device) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "_pin_memory :",
      " self=",
      habana::to_string(self),
      " device=",
      habana::to_string(device));
  HABANA_ASSERT(device.has_value());
  return habana::eager::pin_memory_hpu(self, *device);
}

bool is_pinned(const at::Tensor& self, ::std::optional<at::Device> device) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "is_pinned :",
      " self=",
      habana::to_string(self),
      " device=",
      habana::to_string(device));
  if (!device.has_value()) {
    return false;
  }
  return habana::eager::is_pinned_hpu(self, *device);
}

} // namespace hpu_wrap
