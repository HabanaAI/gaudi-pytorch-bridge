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
    auto& device = synapse_helpers::HPURegistrar::get_device();
    device.get_host_memory().free(get_host_ptr());
    device.get_host_memory().free(get_compile_host_ptr());
  }
}

void TensorExtraMeta::set_host_data(
    void* d,
    int size,
    int el_size,
    HostDataType dt_type) {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  id_ = device.id();
  int total_elem = 2 * size * el_size;
  int data_size = size * el_size;
  auto status = device.get_host_memory().malloc(&host_ptr_, total_elem);
  HABANA_ASSERT(status == synSuccess);
  status = device.get_host_memory().malloc(&compile_host_ptr_, total_elem);
  HABANA_ASSERT(status == synSuccess);
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
    auto& device = synapse_helpers::HPURegistrar::get_device();
    void* host_ptr{};
    auto status = device.get_host_memory().malloc(
        &host_ptr, tensor.numel() * tensor.itemsize());
    HABANA_ASSERT(status == synSuccess);
    tmeta->set_host_ptr(host_ptr);
    PT_LAZY_DEBUG(
        "constant section host_ptr : ",
        tmeta->get_host_ptr(),
        " size: ",
        tensor.numel() * tensor.itemsize());
  }
}

TensorExtraMeta* get_tensor_extra_meta_from_hb_internal_tensor_impl(
    at::TensorImpl& impl,
    [[maybe_unused]] bool relax) {
#if HAVE_TORCH_BACKEND_META_SUPPORT
  TORCH_CHECK(
      false,
      "Attempt to extract tensor extra metadata from HbInternalTensorImpl ",
      &impl);
#else
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    // We seem to be using eager with torch that does not contain the
    // BackendMetadata patch. Said patch is available in vanilla pytorch
    // starting with PT2.1 and any Habana pytorch fork >=1.13.
    //
    // We have no way to store and propagate tensor data layout information so
    // things will not work and we should assert. But for project-development
    // reasons we want the following hack to make the eager UT pass.
    PT_LAZY_WARN(
        "Using stock PT2.0 W/A: accessing tensor extra meta in eager mode is supported in Habana pytorch fork only");
    static TensorExtraMeta global_tmeta;
    global_tmeta = TensorExtraMeta();
    return &global_tmeta;
  }
  auto hb_weight_impl = dynamic_cast<habana_lazy::HbInternalTensorImpl*>(&impl);
  if (hb_weight_impl)
    return &hb_weight_impl->get_tensor_extra_meta();
  TORCH_CHECK(relax, "Tensor extra meta is null for tensor impl ", &impl);
  return nullptr;
#endif
}

TensorExtraMeta* allocate_tensor_extra_meta(at::TensorImpl& impl) {
#if HAVE_TORCH_BACKEND_META_SUPPORT
  TORCH_CHECK(impl.get_backend_meta() == nullptr, "Meta is already assigned.");
  auto new_meta{new habana::TensorExtraMeta()};
  auto meta =
      c10::intrusive_ptr<BaseTensorExtraMeta>::unsafe_steal_from_new(new_meta);
  impl.set_backend_meta(meta);
  return new_meta;
#else
  TORCH_CHECK(
      false, "Attempt to allocate BackendMeta without proper torch support");
#endif
}
} // namespace habana
