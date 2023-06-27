/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "device_type_util.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"

bool isGaudi() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  return habana::HPURegistrar::get_device().type() == synDeviceGaudi;
}

bool isGaudi2() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  return habana::HPURegistrar::get_device().type() == synDeviceGaudi2;
}

bool isGaudi3() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  return habana::HPURegistrar::get_device().type() == synDeviceGaudi3;
}