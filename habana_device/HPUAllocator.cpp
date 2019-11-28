#include "HPUAllocator.h"
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
    if (status != synStatus::synSuccess) {
      VLOG(1) << "synDeviceMalloc failed for " << num_bytes << " bytes.";
    }
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

HabanaAllocator habana_allocator;

// static void HabanaHostDeleter(void* ptr) {
//   habana_allocator.free(ptr);
// }
static void HabanaDeviceDeleter(void* ptr) {
  habana_allocator.free(ptr);
}

// struct HPUHostAllocator final : public at::Allocator {
//   at::DataPtr allocate(size_t size) const override {
//     void* ptr = nullptr;
//     // TODO: get device id

//     if (size != 0) {
//       std::unique_lock<std::mutex> lock(allocation_lock_);
//       auto status{synDeviceMalloc(device_id_, 0, 0, 0, &ptr)};
//       if (status != synStatus::synSuccess)
//         VLOG(1) << "synDeviceMalloc failed for " << size << " bytes.";
//     }

//     return {
//         ptr, ptr, &HabanaHostDeleter, Device(DeviceType::HABANA,
//         device_id_)};
//   }
//   at::DeleterFnPtr raw_deleter() const override {
//     return &HabanaHostDeleter;
//   }
// }; // namespace habana

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