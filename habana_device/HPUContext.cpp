#include "HPUContext.h"
#include "HPUAllocator.h"

namespace at {
namespace habana {

static HPUDeviceAllocator hpu_device_allocator;
// static HPUHostAllocator hpu_host_allocator;

at::Allocator* getHABANADeviceAllocator() {
  return &hpu_device_allocator;
}
// at::Allocator* getHABANAHostAllocator() {
//   return &hpu_host_allocator;
// }

} // namespace habana
} // namespace at
