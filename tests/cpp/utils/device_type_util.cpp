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
#include "device_type_util.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"

bool isGaudi() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  return habana::HPUDeviceContext::get_device().type() == synDeviceGaudi;
}

bool isGaudi2() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  return habana::HPUDeviceContext::get_device().type() == synDeviceGaudi2;
}

bool isGaudi3() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  return habana::HPUDeviceContext::get_device().type() == synDeviceGaudi3;
}