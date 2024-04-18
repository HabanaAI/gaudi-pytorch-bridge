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
#include "dynamic_shape_supported_on_device.h"
#include "backend/habana_device/HPUGuardImpl.h"

bool IsDynamicShapeSupportedOnCurrentDevice() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  // Dynamic shapes are not supported on Gaudi3
  return habana::HPURegistrar::get_device().type() != synDeviceGaudi3;
}
