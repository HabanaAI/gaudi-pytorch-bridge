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
#include "common/utils.h"
#include "backend/backend_meta.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy/hpu_lazy_tensors.h"

namespace common {
void* GetDataPtrFromTensor(const at::Tensor& tensor) {
  return habana_lazy::HbLazyTensor::lazyTensorDataPtr(tensor);
}

bool IsStepMarkerSupported() {
  return true;
}

LibraryType getLoadedLibraryType() {
  return LibraryType::LAZY;
}

bool IsInt64Supported() {
  return GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT);
}

bool IsRecordStreamEnabled() {
  return false;
}

bool IsRecordStreamNoHolderEnabled() {
  return false;
}

} // namespace common
