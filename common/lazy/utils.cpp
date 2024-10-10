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
