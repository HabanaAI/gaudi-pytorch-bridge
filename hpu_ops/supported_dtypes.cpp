/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "supported_dtypes.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "common/utils.h"
#include "habana_kernels/fallback_helper.h"

namespace habana {
SupportedDtypes::SupportedDtypes(std::unordered_set<at::ScalarType> dtypes)
    : m_dtypes(std::move(dtypes)) {}

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
