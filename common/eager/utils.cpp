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
#include <synapse_api_types.h>
#include "backend/backend_meta.h"
#include "backend/create_pt_tensor.h"
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging_pt.h"

namespace common {
void* GetDataPtrFromTensor(const at::Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.storage().data_ptr().get());
}

bool IsStepMarkerSupported() {
  return false;
}

LibraryType getLoadedLibraryType() {
  return LibraryType::EAGER;
}
} // namespace common

namespace habana {

std::string ToString(const StorageExtraMetaMap& map) {
  std::ostringstream sstr;
  sstr << "StorageExtraMeta<offset, permutes>[";
  for (auto it = map.begin(); it != map.end(); ++it) {
    sstr << (it != map.begin() ? ", " : "") << "<" << it->first << ", "
         << VecToString(it->second.get_memory_permutation()) << ">";
  }
  sstr << "]";
  return sstr.str();
}

StorageExtraMeta* get_storage_extra_meta(const at::Tensor& tensor, bool) {
  HABANA_ASSERT(
      tensor.device().type() == c10::DeviceType::HPU,
      "StorageExtraMeta available only on HPU Tensors.");
  if (tensor.unsafeGetTensorImpl()->has_storage()) {
    auto hpu_device_ctx = reinterpret_cast<habana::HPUAllocationContext*>(
        tensor.unsafeGetTensorImpl()->storage().data_ptr().get_context());
    HABANA_ASSERT(
        hpu_device_ctx, "Missing HPUAllocationContext inside storage");
    if (tensor.nbytes() > hpu_device_ctx->num_bytes) {
      // It's possible for i.e. as_strided() called with bigger size than the
      // original buffer.
      PT_EAGER_WARN(
          "Retriving StorageExtraMeta for tensor with nbytes(",
          tensor.nbytes(),
          ") which is more than num_bytes allocated(",
          hpu_device_ctx->num_bytes,
          ") for HPUAllocationContext ",
          hpu_device_ctx,
          " and data_ptr: ",
          tensor.data_ptr());
    }
    if (tensor.nbytes() < hpu_device_ctx->num_bytes) {
      StorageExtraMeta* ptr =
          &(hpu_device_ctx->meta_map[tensor.storage_offset()]);
      PT_EAGER_DEBUG(
          "Accessing HPUAllocationContext ",
          hpu_device_ctx,
          " and StorageExtraMeta ",
          ptr,
          " for offset: ",
          tensor.storage_offset(),
          " with ",
          ToString(hpu_device_ctx->meta_map));
      return ptr;
    } else {
      if (tensor.storage_offset() != 0) {
        PT_EAGER_DEBUG(
            "There should be no offset for accessing StorageMeta of tensor",
            " allocated for >= num_bytes, but it is possible for i.e. as_strided() op.");
      }
      PT_EAGER_DEBUG(
          "Accessing HPUAllocationContext ",
          hpu_device_ctx,
          " and base StorageExtraMeta ",
          &hpu_device_ctx->base_meta,
          " with ",
          ToString(hpu_device_ctx->meta_map));
      return &hpu_device_ctx->base_meta;
    }
  } else {
    return nullptr;
  }
}

void HPUDeviceAllocator_deleter(void* ptr) {
  auto& device =
      HPURegistrar::get_device(HPUDeviceAllocator::allocator_active_device_id);
  auto hpu_device_ctx = reinterpret_cast<HPUAllocationContext*>(ptr);
  auto status{device.get_device_memory().free(hpu_device_ctx->data_ptr)};
  TORCH_HABANA_CHECK(status, "Device Free failed");
  delete hpu_device_ctx;
}

at::DataPtr HPUDeviceAllocator_DataPtr(void* v_ptr, size_t num_bytes) {
  auto ctx = new HPUAllocationContext;
  ctx->data_ptr = v_ptr;
  ctx->num_bytes = num_bytes;
  PT_EAGER_DEBUG(
      "Created HPUAllocationContext ",
      ctx,
      " for data_ptr ",
      ctx->data_ptr,
      " and num_bytes ",
      ctx->num_bytes);
  return {
      v_ptr,
      ctx,
      &HPUDeviceAllocator_deleter,
      at::Device(
          at::DeviceType::HPU, HPUDeviceAllocator::allocator_active_device_id)};
}

} // namespace habana
