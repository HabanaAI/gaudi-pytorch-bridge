/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/backend_meta.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/get_n_bytes.h"
#if HAVE_TORCH_BACKEND_META_SUPPORT
// detecting that there is a torch patch in place that introduces
// c10::BackendMeta in the TensorImpl and we don't have to rely on
// the HbInternalTensorImpl to store the metadata.
#else
#include "habana_lazy/tensor_impl.h"
#endif

namespace habana {

TensorExtraMeta::~TensorExtraMeta() {
  if (get_host_ptr()) {
    auto& device = HPURegistrar::get_device();
    device.get_host_memory().free(get_host_ptr());
    device.get_host_memory().free(get_compile_host_ptr());
  }
}

void TensorExtraMeta::set_host_data(
    void* d,
    int size,
    int el_size,
    HostDataType dt_type) {
  auto& device = HPURegistrar::get_device();
  id_ = device.id();
  int total_elem = 2 * size * el_size;
  int data_size = size * el_size;
  auto status = device.get_host_memory().malloc(&host_ptr_, total_elem);
  HABANA_ASSERT(status == synSuccess, Logger::synStatusToStr(status));
  status = device.get_host_memory().malloc(&compile_host_ptr_, total_elem);
  HABANA_ASSERT(status == synSuccess, Logger::synStatusToStr(status));
  memcpy(host_ptr_, d, data_size);
  char* ptr = static_cast<char*>(host_ptr_) + data_size;
  memcpy(ptr, d, data_size);
  memcpy(
      static_cast<char*>(compile_host_ptr_),
      static_cast<char*>(host_ptr_),
      total_elem);

  total_elem_ = total_elem;
  size_ = size;
  el_size_ = el_size;
  dt_type_ = dt_type;
}

void TensorExtraMeta::update_host_data(
    void* d,
    int size,
    int el_size,
    bool compile) {
  int data_size = size * el_size;
  int total_elem = 2 * data_size;
  memcpy(host_ptr_, d, data_size);
  char* ptr = static_cast<char*>(host_ptr_) + data_size;
  memcpy(ptr, d, data_size);

  if (compile) {
    memcpy(
        static_cast<char*>(compile_host_ptr_),
        static_cast<char*>(host_ptr_),
        total_elem);
  }
}

void TensorExtraMeta::set_const_tensor(
    const at::Tensor& tensor,
    bool is_const_tensor,
    bool relax) {
  auto tmeta{get_tensor_extra_meta(tensor, relax)};
  if (tmeta == nullptr)
    return;

  tmeta->set_is_const_tensor(is_const_tensor);
  PT_LAZY_DEBUG(
      "constant section host_ptr : ",
      tmeta->get_host_ptr(),
      " size: ",
      tensor.numel() * tensor.itemsize(),
      " is_const_tensor_ : ",
      is_const_tensor);
  if (is_const_tensor && (tmeta->get_host_ptr() == nullptr)) {
    auto& device = HPURegistrar::get_device();
    void* host_ptr{};
    auto status = device.get_host_memory().malloc(
        &host_ptr, tensor.numel() * tensor.itemsize());
    HABANA_ASSERT(status == synSuccess, Logger::synStatusToStr(status));
    tmeta->set_host_ptr(host_ptr);
    std::atomic<bool> copyDone{false};
    auto syn_error = device.copy_data_to_host(
        reinterpret_cast<synapse_helpers::device_ptr>(tensor.data_ptr()),
        tmeta->get_host_ptr(),
        reinterpret_cast<synapse_helpers::device_ptr>(
            tensor.storage().data_ptr().get()),
        habana_helpers::GetNBytes(tensor),
        [&copyDone]() { copyDone = true; },
        true);
    TORCH_HABANA_CHECK(syn_error.status, syn_error.error);
    // wait for copy completion
    while (!copyDone) {
      std::this_thread::yield();
    }
    tmeta->set_data_in_host_memory(true);
    PT_LAZY_DEBUG(
        "constant section host_ptr : ",
        tmeta->get_host_ptr(),
        " size: ",
        tensor.numel() * tensor.itemsize());
  }
}

TensorExtraMeta* allocate_tensor_extra_meta(at::TensorImpl& impl) {
  TORCH_CHECK(impl.get_backend_meta() == nullptr, "Meta is already assigned.");
  auto new_meta{new habana::TensorExtraMeta()};
  auto meta =
      c10::intrusive_ptr<BaseTensorExtraMeta>::unsafe_steal_from_new(new_meta);
  impl.set_backend_meta(meta);
  return new_meta;
}

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

StorageExtraMeta* get_storage_extra_meta(
    const c10::TensorImpl* tensor_impl,
    at::optional<size_t> nbytes) {
  HABANA_ASSERT(
      tensor_impl->device().type() == c10::DeviceType::HPU,
      "StorageExtraMeta available only on HPU Tensors.");
  if (tensor_impl->has_storage()) {
    auto alloc_ctx = reinterpret_cast<habana::HPUAllocationContext*>(
        tensor_impl->storage().data_ptr().get_context());
    if (!alloc_ctx) {
      // i.e. when allocation is 0 bytes, we do not create HPUAllocationContext
      PT_BRIDGE_DEBUG(
          "Trying to get StorageExtraMeta from TensorImpl ",
          tensor_impl,
          " without an Allocation Context. Returning nullptr..");
      return nullptr;
    }
    if (nbytes.has_value() && nbytes.value() > alloc_ctx->num_bytes) {
      // It's possible for i.e. as_strided() called with bigger size than the
      // original buffer.
      PT_BRIDGE_DEBUG(
          "Retrieving StorageExtraMeta for tensor with nbytes(",
          nbytes.value(),
          ") which is more than num_bytes allocated(",
          alloc_ctx->num_bytes,
          ") for HPUAllocationContext ",
          alloc_ctx,
          " and data_ptr: ",
          tensor_impl->data());
    }
    if (nbytes.has_value() && nbytes.value() < alloc_ctx->num_bytes) {
      // It is intended to access the map with [], as we always want to get an
      // entry for new storage offset (either create or lookup is fine)
      // TODO: Add assert for overlapping views
      StorageExtraMeta* ptr =
          &(alloc_ctx->meta_map[tensor_impl->storage_offset()]);
      PT_BRIDGE_DEBUG(
          "Accessing HPUAllocationContext ",
          alloc_ctx,
          " and StorageExtraMeta ",
          ptr,
          " for offset: ",
          tensor_impl->storage_offset(),
          " with ",
          ToString(alloc_ctx->meta_map));
      return ptr;
    } else {
      if (tensor_impl->storage_offset() != 0) {
        // There should be no offset for accessing StorageMeta of tensor
        // allocated for >= num_bytes, but it's possible for i.e. as_strided op.
        PT_BRIDGE_DEBUG(
            "Non-zero storage_offset(",
            tensor_impl->storage_offset(),
            ") when accessing base StorageExtraMeta.");
      }
      PT_BRIDGE_DEBUG(
          "Accessing HPUAllocationContext ",
          alloc_ctx,
          " and base StorageExtraMeta ",
          &alloc_ctx->base_meta,
          " with ",
          ToString(alloc_ctx->meta_map));
      return &alloc_ctx->base_meta;
    }
  } else {
    PT_BRIDGE_DEBUG(
        "No StorageExtraMeta available - TensorImpl has no storage. Returning nullptr..");
    return nullptr;
  }
}

StorageExtraMeta* get_storage_extra_meta(const at::Tensor& tensor) {
  PT_BRIDGE_DEBUG(
      "Getting StorageExtraMeta for tensor : ",
      tensor.toString(),
      ", sizes: ",
      tensor.sizes());
  return get_storage_extra_meta(tensor.unsafeGetTensorImpl(), tensor.nbytes());
}

} // namespace habana
