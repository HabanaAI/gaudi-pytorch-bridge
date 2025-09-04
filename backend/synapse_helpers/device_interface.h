/**
 * Copyright (c) 2025 Intel Corporation
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
#include <synapse_api_types.h>

namespace synapse_helpers {

class device_interface {
 public:
  virtual ~device_interface() = default;

  // Device id as assigned by Synapse during device acquire.
  // Note this is unrelated to module ID, e.g., multiple devices acquired in
  // different processes can have the same ID here.
  virtual synDeviceId id() const = 0;

  // Informs whether host memory allocations are cached.
  virtual bool HostMemoryCacheEnabled() const = 0;
};

} // namespace synapse_helpers
