/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include "PinnedMemoryAllocator.h"
#include "habana_helpers/logging.h"
#include "hpu_cached_devices.h"

namespace habana {
synDeviceId PinnedMemoryAllocator::allocator_active_device_id = -1;

static PinnedMemoryAllocator pin_memory_allocator;
at::Allocator* getPinnedMemoryAllocator() {
  return &pin_memory_allocator;
}

bool PinnedMemoryAllocator_is_pinned(void* ptr) {
  auto& device = HPURegistrar::get_device(
      habana::PinnedMemoryAllocator::allocator_active_device_id);
  return device.get_host_memory().is_host_memory(ptr);
}

PinnedMemoryAllocator::PinnedMemoryAllocator() = default;
PinnedMemoryAllocator::~PinnedMemoryAllocator() = default;

void PinnedMemoryAllocator::deleter(void* ptr) {
  auto& device = HPURegistrar::get_device(
      habana::PinnedMemoryAllocator::allocator_active_device_id);
  device.get_host_memory().free(ptr);
}

#if IS_PYTORCH_AT_LEAST(2, 3)
at::DataPtr PinnedMemoryAllocator::allocate(size_t size) {
#else
at::DataPtr PinnedMemoryAllocator::allocate(size_t size) const {
#endif
  void* ptr = nullptr;
  if (size != 0) {
    auto& device = HPURegistrar::get_device(
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

#if IS_PYTORCH_AT_LEAST(2, 3)
void PinnedMemoryAllocator::copy_data(
    [[maybe_unused]] void* dest,
    [[maybe_unused]] const void* src,
    [[maybe_unused]] std::size_t count) const {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "Not implemented for PinnedMemoryAllocator");
}
#endif
} // namespace habana
