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
#include "dtype_supported_on_device.h"
#include "backend/habana_device/HPUGuardImpl.h"

bool IsDtypeSupportedOnCurrentDevice(torch::ScalarType dtype) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPUDeviceContext::get_device();
  switch (device.type()) {
    case synDeviceGaudi:
      switch (dtype) {
        case torch::kFloat16:
          return false;
        default:
          break;
      }
      break;
    default:
      break;
  }
  return true;
}
