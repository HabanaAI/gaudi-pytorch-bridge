/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "PinnedMemoryAllocator.h"
#include "HPUCheck.h"
#include "hpu_cached_devices.h"

namespace at {
namespace habana {
synDeviceId PinnedMemoryAllocator::allocator_active_device_id = -1;

static PinnedMemoryAllocator pin_memory_allocator;
at::Allocator* getPinnedMemoryAllocator() {
  return &pin_memory_allocator;
}

bool PinnedMemoryAllocator_is_pinned(void* ptr) {
  auto& device = synapse_helpers::HPURegistrar::get_device(
      habana::PinnedMemoryAllocator::allocator_active_device_id);
  return device.get_host_memory().is_host_memory(ptr);
}

PinnedMemoryAllocator::PinnedMemoryAllocator() = default;
PinnedMemoryAllocator::~PinnedMemoryAllocator() = default;

void PinnedMemoryAllocator::deleter(void* ptr) {
  auto& device = synapse_helpers::HPURegistrar::get_device(
      habana::PinnedMemoryAllocator::allocator_active_device_id);
  device.get_host_memory().free(ptr);
}

at::DataPtr PinnedMemoryAllocator::allocate(size_t size) const {
  void* ptr;
  if (size != 0) {
    auto& device = synapse_helpers::HPURegistrar::get_device(
        habana::PinnedMemoryAllocator::allocator_active_device_id);
    auto status = device.get_host_memory().malloc(&ptr, size);
    TORCH_HABANA_CHECK(
        status, "synHostMalloc failed to allocate ", size, " bytes");
  }
  return {ptr, ptr, &PinnedMemoryAllocator::deleter, Device(DeviceType::CPU)};
}

at::DeleterFnPtr PinnedMemoryAllocator::raw_deleter() const {
  return &PinnedMemoryAllocator::deleter;
}

} // namespace habana
} // namespace at
