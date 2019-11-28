#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "synapse/include/synapse_api.h"

namespace at {
namespace habana {

void* HabanaAllocator::malloc(size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }

  uint64_t ptr{0};
  {
    std::unique_lock<std::mutex> lock(allocation_lock_);
    auto status{synDeviceMalloc(device_id_, num_bytes, 0, 0, &ptr)};
    TORCH_HABANA_CHECK(
        status, "synDeviceMalloc failed to allocate ", num_bytes, " bytes");
  }

  void* v_ptr = reinterpret_cast<void*>(ptr);
  return v_ptr;
}

void HabanaAllocator::free(void* ptr) {
  if (!ptr) {
    return;
  }
  uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
  std::unique_lock<std::mutex> lock(allocation_lock_);
  auto status{synDeviceFree(device_id_, ptr_address, 0)};
  VLOG(1) << "HabanaAllocator::Free for " << std::hex << ptr_address;
  if (status != synStatus::synSuccess) {
    VLOG(1) << "synDeviceFree failed for " << std::hex << ptr_address;
  }
}

static HabanaAllocator habana_allocator;

static void HabanaDeviceDeleter(void* ptr) {
  habana_allocator.free(ptr);
}

at::DataPtr HPUDeviceAllocator::allocate(size_t size) const {
  // TODO: get device id
  void* ptr = habana_allocator.malloc(size);

  return {
      ptr, ptr, &HabanaDeviceDeleter, Device(DeviceType::HABANA, device_id_)};
}

at::DeleterFnPtr HPUDeviceAllocator::raw_deleter() const {
  return &HabanaDeviceDeleter;
}

} // namespace habana
} // namespace at