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
} // namespace common
