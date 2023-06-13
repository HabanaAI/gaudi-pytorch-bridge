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
#include <synapse_api_types.h>
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/hpu_cached_devices.h"
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

namespace habana {
StorageExtraMeta* get_storage_extra_meta(const at::Tensor& tensor, bool relax) {
  auto tmeta = get_tensor_extra_meta(tensor, relax);
  if (tmeta) {
    return &(tmeta->storage_meta_);
  } else {
    return nullptr;
  }
}

void HPUDeviceAllocator_deleter(void* ptr) {
  auto& device =
      HPURegistrar::get_device(HPUDeviceAllocator::allocator_active_device_id);
  auto status{device.get_device_memory().free(ptr)};
  TORCH_HABANA_CHECK(status, "Device Free failed");
}

at::DataPtr HPUDeviceAllocator_DataPtr(void* v_ptr, size_t) {
  return {
      v_ptr,
      v_ptr,
      &HPUDeviceAllocator_deleter,
      at::Device(
          at::DeviceType::HPU, HPUDeviceAllocator::allocator_active_device_id)};
}

} // namespace habana
