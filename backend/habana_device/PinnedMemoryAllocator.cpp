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
#include "PinnedMemoryAllocator.h"
#include "HPUDevice.h"
#include "habana_helpers/logging.h"

namespace habana {
synDeviceId PinnedMemoryAllocator::allocator_active_device_id = -1;

static PinnedMemoryAllocator pin_memory_allocator;
at::Allocator* PinnedMemoryAllocator_get() {
  return &pin_memory_allocator;
}

bool PinnedMemoryAllocator_is_pinned(const void* ptr) {
  auto& device = HPUDeviceContext::get_device(
      habana::PinnedMemoryAllocator::allocator_active_device_id);
  return device.get_host_memory().is_host_memory(const_cast<void*>(ptr));
}

PinnedMemoryAllocator::PinnedMemoryAllocator() = default;
PinnedMemoryAllocator::~PinnedMemoryAllocator() = default;

void PinnedMemoryAllocator::deleter(void* ptr) {
  auto& device = HPUDeviceContext::get_device(
      habana::PinnedMemoryAllocator::allocator_active_device_id);
  device.get_host_memory().free(ptr);
}

at::DataPtr PinnedMemoryAllocator::allocate(size_t size) {
  void* ptr = nullptr;
  if (size != 0) {
    auto& device = HPUDeviceContext::get_device(
        habana::PinnedMemoryAllocator::allocator_active_device_id);
    auto status = device.get_host_memory().malloc(&ptr, size);
    TORCH_HABANA_CHECK(
        status, "synHostMalloc failed to allocate ", size, " bytes");
  }
  return {
      ptr,
      ptr,
      &PinnedMemoryAllocator::deleter,
      at::Device(at::DeviceType::CPU)};
}

at::DeleterFnPtr PinnedMemoryAllocator::raw_deleter() const {
  return &PinnedMemoryAllocator::deleter;
}

void PinnedMemoryAllocator::copy_data(
    [[maybe_unused]] void* dest,
    [[maybe_unused]] const void* src,
    [[maybe_unused]] std::size_t count) const {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "Not implemented for PinnedMemoryAllocator");
}
} // namespace habana
