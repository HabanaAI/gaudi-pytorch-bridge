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
#include "dtype_supported_on_device.h"
#include "backend/habana_device/HPUGuardImpl.h"

bool IsDtypeSupportedOnCurrentDevice(torch::ScalarType dtype) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
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
