/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "backend/habana_device/HPUHooksInterface.h"
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/random.h"
namespace habana {

void HPUHooks::init() const {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
}

const at::Generator& HPUHooks::getDefaultHPUGenerator(at::DeviceIndex) const {
  return detail::getDefaultHPUGenerator();
}

bool HPUHooks::hasHPU() const {
  return true;
}

at::Device HPUHooks::getDeviceFromPtr(void*) const {
  // TODO add check if pointer valid
  habana::HABANAGuardImpl device_guard;
  return device_guard.getDevice();
}

bool HPUHooks::isPinnedPtr(const void* data) const {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  return PinnedMemoryAllocator_is_pinned(data);
}

at::Allocator* HPUHooks::getPinnedMemoryAllocator() const {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
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
