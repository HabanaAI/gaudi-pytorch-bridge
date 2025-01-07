/**
* Copyright (c) 2021-2024 Intel Corporation
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
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <synapse_api_types.h>
#include "backend/synapse_helpers/device.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {

at::Allocator* PinnedMemoryAllocator_get();
bool PinnedMemoryAllocator_is_pinned(const void* ptr);

class PinnedMemoryAllocator final : public at::Allocator {
 public:
  PinnedMemoryAllocator();
  ~PinnedMemoryAllocator();
  at::DeleterFnPtr raw_deleter() const override;
  static void deleter(void* ptr);

  at::DataPtr allocate(size_t size) override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;
  // user must manually set active device before calling allocator functions
  static synDeviceId allocator_active_device_id;
};

} // namespace habana
