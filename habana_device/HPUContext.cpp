#include "HPUContext.h"
#include "HPUAllocator.h"

namespace at {
namespace habana {

static HPUDeviceAllocator hpu_device_allocator;

at::Allocator* getHABANADeviceAllocator() {
  return &hpu_device_allocator;
}

// TODO: it might be not the best place to put this macro. I am confused how
// allocators are registered.
REGISTER_ALLOCATOR(DeviceType::HABANA, &at::habana::hpu_device_allocator);

} // namespace habana
} // namespace at
