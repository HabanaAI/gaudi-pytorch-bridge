/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <synapse_api.h>

#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "HPUGuardImpl.h"
#include "habana_helpers/logging.h"
#include "hpu_cached_devices.h"

namespace at {
namespace habana {

synDeviceId HPUDeviceAllocator::allocator_active_device_id = -1;

static HPUDeviceAllocator hpu_device_allocator;

at::Allocator* getHABANADeviceAllocator() {
  return &hpu_device_allocator;
}

// TODO: it might be not the best place to put this macro. I am confused how
// allocators are registered.
REGISTER_ALLOCATOR(DeviceType::HABANA, &at::habana::hpu_device_allocator);
} // namespace habana

namespace detail {

C10_REGISTER_GUARD_IMPL(HABANA, HABANAGuardImpl);

} // namespace detail

namespace habana {

HPUAllocator::HPUAllocator(uint32_t device) : device_id(device) {}

void HPUAllocator::reset() {
  PT_DEVICE_WARN("HPUAllocator::reset should not be invoked.");
}

void HPUAllocator::release() {
  device_id = synapse_helpers::device::INVALID_ID;
  PT_DEVICE_WARN("HPUAllocator::release should not be invoked.");
}

void* HPUAllocator::alloc(size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }

  uint64_t ptr{0};
  auto status{synDeviceMalloc(device_id, num_bytes, 0, 0, &ptr)};
  TORCH_HABANA_CHECK(
      status, "synDeviceMalloc failed to allocate ", num_bytes, " bytes");

  void* v_ptr = reinterpret_cast<void*>(ptr);
  return v_ptr;
}

void HPUAllocator::free(void* ptr) {
  if (nullptr == ptr) {
    return;
  }
  uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
  auto status{synDeviceFree(device_id, ptr_address, 0)};
  TORCH_HABANA_CHECK(status, "synDeviceFree failed");
}

HPUDeviceAllocator::HPUDeviceAllocator() = default;

void HPUDeviceAllocator::deleter(void* ptr) {
  if (nullptr == ptr) {
    return;
  }
  uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");
  auto status{synDeviceFree(allocator_active_device_id, ptr_address, 0)};
  TORCH_HABANA_CHECK(status, "synDeviceFree failed");
}

at::DataPtr HPUDeviceAllocator::allocate(size_t size) const {
  size_t num_bytes = size;
  uint64_t ptr{0};
  if (num_bytes != 0) {
    TORCH_CHECK(
        habana::HPUDeviceAllocator::allocator_active_device_id == 0,
        "habana active device: ",
        habana::HPUDeviceAllocator::allocator_active_device_id,
        " != 0");
    auto status{
        synDeviceMalloc(allocator_active_device_id, num_bytes, 0, 0, &ptr)};
    TORCH_HABANA_CHECK(
        status, "synDeviceMalloc failed to allocate ", num_bytes, " bytes");
  }

  void* v_ptr = reinterpret_cast<void*>(ptr);
  TORCH_CHECK(
      habana::HPUDeviceAllocator::allocator_active_device_id == 0,
      "habana active device: ",
      habana::HPUDeviceAllocator::allocator_active_device_id,
      " != 0");
  return {v_ptr,
          v_ptr,
          &HPUDeviceAllocator::deleter,
          Device(DeviceType::HABANA, allocator_active_device_id)};
}

at::DeleterFnPtr HPUDeviceAllocator::raw_deleter() const {
  return &HPUDeviceAllocator::deleter;
}

} // namespace habana
} // namespace at

namespace synapse_helpers {

HPURegistrar& HPURegistrar::get_hpu_registrar() {
  static HPURegistrar* instance = new HPURegistrar();
  return *instance;
}

} // namespace synapse_helpers

void print_live_allocations() {
  std::cout << "\nNo log for device memory allocation is collected. Use the following commands "
               "to enable allocation tracking and reporting with hb_torch.memstat_livealloc() - \n"
               "HBN_SYNAPSE_LOGGER_COMMANDS=log_device_alloc "
               "LD_PRELOAD=$BUILD_ROOT_LATEST/pytorch_synapse_logger.so a.out...\n\n";
}
