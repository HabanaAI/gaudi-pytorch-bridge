#pragma once
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <mutex>

namespace at {
namespace habana {

class HabanaAllocator {
  int device_id_ = 0; // TODO:
  std::mutex allocation_lock_;

 public:
  void* malloc(size_t num_bytes);
  void free(void* ptr);
};

// class HPUHostAllocator final : public at::Allocator {
//   std::mutex allocation_lock_;

//  public:
//   at::DataPtr allocate(size_t size) const override;
//   at::DeleterFnPtr raw_deleter() const override;
//   int device_id_ = 0; // TODO:
// };

class HPUDeviceAllocator final : public at::Allocator {
  std::mutex allocation_lock_;

 public:
  at::DataPtr allocate(size_t size) const override;
  at::DeleterFnPtr raw_deleter() const override;
  int device_id_ = 0; // TODO:
};

} // namespace habana
} // namespace at
