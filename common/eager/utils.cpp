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
#include <ATen/Tensor.h>
#include "backend/habana_device/HPUDevice.h"
#include "backend/synapse_helpers/env_flags.h"

namespace common {
void* GetDataPtrFromTensor(const at::Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.storage().data_ptr().get());
}

bool IsStepMarkerSupported() {
  return false;
}

bool IsInt64Supported() {
  // In eager we want to use flag only if it is defined, if not - return true,
  // because it is the default value for eager mode.
  // Until issues with failing tests are resolved result shall remain false
  auto result = true;

  if (habana::HPUDeviceContext::get_device().name() == "GAUDI")
    result = false;

  if (IS_ENV_FLAG_DEFINED_NEW(PT_ENABLE_INT64_SUPPORT)) {
    result = GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT);
  }
  return result;
}

LibraryType getLoadedLibraryType() {
  return LibraryType::EAGER;
}

namespace {
bool _IsRecordStreamEnabled() {
  bool value = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_RECORD_STREAM);
  return value;
}

bool _IsRecordStreamNoHolderEnabled() {
  bool value = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_RECORD_STREAM_NOHOLDER);
  return value and _IsRecordStreamEnabled();
}

thread_local bool _recordStreamEnabled = _IsRecordStreamEnabled();
thread_local bool _recordStreamNoHolderEnabled =
    _IsRecordStreamNoHolderEnabled();

} // namespace

bool IsRecordStreamEnabled() {
  return _recordStreamEnabled;
}

bool IsRecordStreamNoHolderEnabled() {
  return _recordStreamNoHolderEnabled;
}

} // namespace common
