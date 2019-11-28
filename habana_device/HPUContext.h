#pragma once
#include <ATen/ATen.h>

namespace at {
namespace habana {

at::Allocator* getHABANADeviceAllocator();

} // namespace habana
} // namespace at
