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
#pragma once
#include <ATen/detail/HPUHooksInterface.h>

namespace habana {

struct HPUHooks : public at::HPUHooksInterface {
  HPUHooks(at::HPUHooksArgs){};

  void init() const override;

  const at::Generator& getDefaultHPUGenerator(
      at::DeviceIndex device_index = -1) const override;

  bool hasHPU() const override;
  at::Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(const void* data) const override;
  at::Allocator* getPinnedMemoryAllocator() const override;
  bool hasPrimaryContext(at::DeviceIndex device_index) const override;
};
} // namespace habana
