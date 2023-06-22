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
#include <pybind11/chrono.h>
#include <torch/extension.h>
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUAllocator.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_lazy/hlexec.h"

namespace {
int GetCurrentThreadDevice() {
  auto& d = habana::HPURegistrar::get_device();
  return d.id();
}
} // namespace

class SharedTensorExtraMeta {
 public:
  auto get_memory_permutation() {
    return get().get_memory_permutation();
  }
  auto get_dont_allow_permutation() const {
    return get().get_dont_allow_permutation();
  }
  const habana::TensorExtraMeta& get() const {
    return tmeta_;
  }
  static std::optional<SharedTensorExtraMeta> create(const at::Tensor& tensor) {
#if HAVE_TORCH_BACKEND_META_SUPPORT
    auto impl{tensor.unsafeGetTensorImpl()};
    c10::intrusive_ptr<habana::BaseTensorExtraMeta> meta{
        impl->get_backend_meta()};
    if (!meta)
      return {};
    auto tmeta_ptr{dynamic_cast<habana::TensorExtraMeta*>(meta.get())};
    PT_EAGER_DEBUG(
        "Producing SharedTensorExtraMeta for impl : ",
        impl,
        " tensor meta at address ",
        tmeta_ptr,
        " storage address : ",
        tensor.data_ptr());

    TORCH_CHECK(
        tmeta_ptr != nullptr,
        "Got BackendMeta ",
        meta.get(),
        " but it is not habana::TensorExtraMeta");
    return std::optional<SharedTensorExtraMeta>(
        SharedTensorExtraMeta(meta, *tmeta_ptr));
#else
    return {};
#endif
  }

 private:
  c10::intrusive_ptr<habana::BaseTensorExtraMeta> tmeta_ref_holder_;
  habana::TensorExtraMeta& tmeta_;

  SharedTensorExtraMeta(
      c10::intrusive_ptr<habana::BaseTensorExtraMeta> tmeta_ref_holder,
      habana::TensorExtraMeta& tmeta)
      : tmeta_ref_holder_{tmeta_ref_holder}, tmeta_{tmeta} {}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_hb_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def(
      "_iter_mark_step", []() { habana_lazy::HbLazyTensor::IterStepMarker(); });
  m.def(
      "_mark_step",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::StepMarkerBind(device_str);
      },
      py::arg("device_str") = "");
  m.def("_get_default_generator", []() {
    return habana::getDefaultHPUGenerator();
  });
  m.doc() = "This module registers hpu lazy api.";
  py::class_<SharedTensorExtraMeta>(m, "TensorExtraMeta")
      .def_property_readonly(
          "memory_permutation", &SharedTensorExtraMeta::get_memory_permutation)
      .def_property_readonly(
          "dont_allow_permutation",
          &SharedTensorExtraMeta::get_dont_allow_permutation);
  m.def("get_tensor_extra_meta", &SharedTensorExtraMeta::create);
}
