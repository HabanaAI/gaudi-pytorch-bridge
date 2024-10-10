/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "supported_dtypes.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "common/utils.h"
#include "habana_kernels/fallback_helper.h"

namespace habana {
SupportedDtypes::SupportedDtypes(
    std::unordered_map<int, std::unordered_set<at::ScalarType>>
        per_gen_dtypes) {
  if (per_gen_dtypes.size() == 1) {
    m_dtypes = std::move(per_gen_dtypes.begin()->second);
    return;
  }

  auto get_curr_dev_type = []() {
    // get device should be invoked in case device has not been initialized yet
    HABANAGuardImpl device_guard;
    device_guard.getDevice();

    auto dev = HPURegistrar::get_device().type();
    return dev;
  };

  const static int curr_dev_type = get_curr_dev_type();
  HABANA_ASSERT(
      per_gen_dtypes.count(curr_dev_type),
      "No dtypes defined for device type ",
      curr_dev_type);

  m_dtypes = std::move(per_gen_dtypes.at(curr_dev_type));
}

bool SupportedDtypes::count(at::ScalarType type) const {
  return m_dtypes.count(type) ||
      (!common::IsInt64Supported() && type == at::ScalarType::Long &&
       m_dtypes.count(at::ScalarType::Int));
}

bool SupportedDtypes::count(const at::Tensor& tensor) const {
  return count(tensor.scalar_type());
}

bool SupportedDtypes::count(const at::optional<at::Tensor>& tensor) const {
  return tensor.has_value() and count(tensor.value());
}
} // namespace habana
