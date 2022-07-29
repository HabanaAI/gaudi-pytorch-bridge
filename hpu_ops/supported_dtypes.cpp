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
#include "reduction_template.h"

namespace habana {
SupportedDtypes::SupportedDtypes(
    std::unordered_map<int, std::unordered_set<at::ScalarType>>
        per_gen_dtypes) {
  const static int curr_dev_type =
      synapse_helpers::HPURegistrar::get_device().type();
  if (per_gen_dtypes.size() == 1) {
    m_dtypes = std::move(per_gen_dtypes.begin()->second);
  } else if (per_gen_dtypes.count(curr_dev_type)) {
    m_dtypes = std::move(per_gen_dtypes.at(curr_dev_type));
  }
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

bool SupportedDtypes::count(
    const at::Tensor& tensor,
    at::optional<at::ScalarType> type) const {
  return m_dtypes.count(get_dtype_from_self(tensor, type, true));
}
} // namespace habana
