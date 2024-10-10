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
#include "common/utils.h"
#include <ATen/Tensor.h>
#include "backend/habana_device/hpu_cached_devices.h"
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

  if (habana::HPURegistrar::get_device().name() == "GAUDI")
    result = false;

  if (IS_ENV_FLAG_DEFINED_NEW(PT_ENABLE_INT64_SUPPORT)) {
    result = GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT);
  }
  return result;
}

LibraryType getLoadedLibraryType() {
  return LibraryType::EAGER;
}

bool IsRecordStreamEnabled() {
  static bool value = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_RECORD_STREAM);
  return value;
}

bool IsRecordStreamNoHolderEnabled() {
  static bool value = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_RECORD_STREAM_NOHOLDER);
  return value and IsRecordStreamEnabled();
}

} // namespace common
