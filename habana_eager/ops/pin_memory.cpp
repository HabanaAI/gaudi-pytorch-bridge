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
  auto* allocator = habana::PinnedMemoryAllocator_get();
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
  HABANA_ASSERT(device.has_value(), "Unable to pin memory to an null device");
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
