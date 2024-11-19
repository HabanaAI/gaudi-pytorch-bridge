/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_device/HPUHooksInterface.h"
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/random.h"
namespace habana {

#if IS_PYTORCH_AT_LEAST(2, 6)
void HPUHooks::init() const {
#else
void HPUHooks::initHPU() const {
#endif
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
}

bool HPUHooks::hasHPU() const {
  return true;
}

const at::Generator& HPUHooks::getDefaultHPUGenerator(at::DeviceIndex) const {
  return detail::getDefaultHPUGenerator();
}

at::Device HPUHooks::getDeviceFromPtr(void*) const {
  // TODO add check if pointer valid
  habana::HABANAGuardImpl device_guard;
  return device_guard.getDevice();
}

bool HPUHooks::isPinnedPtr(const void* data) const {
  return PinnedMemoryAllocator_is_pinned(data);
}

at::Allocator* HPUHooks::getPinnedMemoryAllocator() const {
  return PinnedMemoryAllocator_get();
}

bool HPUHooks::hasPrimaryContext(at::DeviceIndex) const {
  // According to interface, this function is used to determine:
  // 'Whether the device at device_index is fully initialized or not.'
  // and for HPU, device index is irrelevant, as single device is supported in
  // process and only check for device acquisition should be enough.
  return HPUDeviceContext::is_device_acquired();
}

using at::HPUHooksRegistry;
using at::RegistererHPUHooksRegistry;
REGISTER_HPU_HOOKS(HPUHooks);

} // namespace habana
