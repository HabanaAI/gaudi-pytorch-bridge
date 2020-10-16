#include <ATen/detail/HabanaHooksInterface.h>
#include <c10/util/Exception.h>
#include <habana_device/HPUGuardImpl.h>
#include <habana_device/PinnedMemoryAllocator.h>
#include <habana_device/hpu_cached_devices.h>
#include <habana_hooks/HabanaHooks.h>
#include "habana_helpers/logging.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>

namespace at {
namespace habana {
namespace detail {

static bool init() {
  at::detail::HABANAGuardImpl device_guard;
  try {
    device_guard.getDevice();
    return true;
  } catch (...) {
    PT_HABANAHOOKS_WARN("Get Device failed");
    return false;
  }
}

bool HabanaHooks::isPinnedPtr(void* data) const {
  return PinnedMemoryAllocator_is_pinned(data);
}

bool HabanaHooks::hasHabana() const {
  return init();
}

int64_t HabanaHooks::current_device() const {
  init();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  int id = device.id();
  PT_HABANAHOOKS_DEBUG("Current Device ID", id);
  return id;
}

Allocator* HabanaHooks::getPinnedMemoryAllocator() const {
  return at::habana::getPinnedMemoryAllocator();
}

int HabanaHooks::getNumGPUs() const {
  int count = 0;
  if (init()) {
    auto& device = synapse_helpers::HPURegistrar::get_device();
    int count = device.get_count();
    PT_HABANAHOOKS_DEBUG("Device count", count);
  }
  return count;
}

// Sigh, the registry doesn't support namespaces :(
using at::HabanaHooksRegistry;
using at::RegistererHabanaHooksRegistry;

REGISTER_HABANA_HOOKS(HabanaHooks);

} // namespace detail
} // namespace habana
} // namespace at
