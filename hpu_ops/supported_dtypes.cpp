/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "supported_dtypes.h"
#include "habana_kernels/fallback_helper.h"
#include "pytorch_helpers/habana_device/hpu_cached_devices.h"

namespace habana {
SupportedDtypes::SupportedDtypes(
    std::unordered_map<int, std::unordered_set<at::ScalarType>>
        per_gen_dtypes) {
  if (per_gen_dtypes.size() == 1) {
    m_dtypes = std::move(per_gen_dtypes.begin()->second);
    return;
  }

  auto get_curr_dev_type = []() {
    auto dev = synapse_helpers::HPURegistrar::get_device().type();
    return dev;
  };

  const static int curr_dev_type = get_curr_dev_type();
  HABANA_ASSERT(
      per_gen_dtypes.count(curr_dev_type),
      "No dtypes defined for device type ",
      curr_dev_type);

  m_dtypes = std::move(per_gen_dtypes.at(curr_dev_type));
}

SupportedDtypes::~SupportedDtypes() {
  static std::once_flag flag;
  std::call_once(
      flag, []() { HpuFallbackHelper::get()->print_fallback_freq(); });
}

bool SupportedDtypes::count(at::ScalarType type) const {
  return m_dtypes.count(type);
}

bool SupportedDtypes::count(const at::Tensor& tensor) const {
  return count(tensor.scalar_type());
}

bool SupportedDtypes::count(const at::optional<at::Tensor>& tensor) const {
  return tensor.has_value() and count(tensor.value());
}
} // namespace habana
