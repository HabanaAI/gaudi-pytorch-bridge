#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "synapse/include/synapse_api.h"

namespace at {
namespace habana {

synDeviceId allocator_active_device_id = -1;

void* HabanaAllocator::malloc(size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }

  uint64_t ptr{0};
  TORCH_CHECK(
      habana::allocator_active_device_id == 0, "habana active device: ", habana::allocator_active_device_id, " != 0");
  auto status{
      synDeviceMalloc(allocator_active_device_id, num_bytes, 0, 0, &ptr)};
  TORCH_HABANA_CHECK(
      status, "synDeviceMalloc failed to allocate ", num_bytes, " bytes");

  void* v_ptr = reinterpret_cast<void*>(ptr);
  return v_ptr;
}

void HabanaAllocator::free(void* ptr) {
  if (!ptr) {
    return;
  }
  uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
  TORCH_CHECK(
      habana::allocator_active_device_id == 0, "habana active device: ", habana::allocator_active_device_id, " != 0");
  auto status{synDeviceFree(allocator_active_device_id, ptr_address, 0)};
  TORCH_HABANA_CHECK(status, "synDeviceFree failed");
}

static HabanaAllocator habana_allocator;

static void HabanaDeviceDeleter(void* ptr) {
  habana_allocator.free(ptr);
}

at::DataPtr HPUDeviceAllocator::allocate(size_t size) const {
  void* ptr = habana_allocator.malloc(size);
  TORCH_CHECK(
      habana::allocator_active_device_id == 0, "habana active device: ", habana::allocator_active_device_id, " != 0");
  return {ptr,
          ptr,
          &HabanaDeviceDeleter,
          Device(DeviceType::HABANA, allocator_active_device_id)};
}

at::DeleterFnPtr HPUDeviceAllocator::raw_deleter() const {
  return &HabanaDeviceDeleter;
}

} // namespace habana
} // namespace at

habana_helpers::HabanaAllocator::HabanaAllocator(uint32_t device)
    : device_id(device) {}

void habana_helpers::HabanaAllocator::reset(uint32_t device) {
  device_id = device;
  TORCH_WARN(
      "You probably shouldn't call HabanaAllocator::reset. If you really need to do it then remove this assert.\nDevice id: ",
      device);
}
void habana_helpers::HabanaAllocator::release() {
  device_id = synapse_helpers::device_handle::INVALID_ID;
  TORCH_WARN(
      "You probably shouldn't call HabanaAllocator::release. If you really need to do it then remove this assert");
}
void* habana_helpers::HabanaAllocator::alloc(size_t num_bytes) {
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
void habana_helpers::HabanaAllocator::free(void* ptr) {
  if (!ptr) {
    return;
  }
  uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
  auto status{synDeviceFree(device_id, ptr_address, 0)};
  TORCH_HABANA_CHECK(status, "synDeviceFree failed");
}