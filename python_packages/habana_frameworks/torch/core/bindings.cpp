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
#include <pybind11/chrono.h>
#include <torch/extension.h>
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUDevice.h"
#include "backend/random.h"
#include "habana_lazy/hpu_lazy_tensors.h"

namespace {
int GetCurrentThreadDevice() {
  auto& d = habana::HPUDeviceContext::get_device();
  return d.id();
}
} // namespace

class SharedTensorExtraMeta {
 public:
  auto set_is_const_tensor(bool is_const_tensor) {
    return tmeta_.set_is_const_tensor(is_const_tensor);
  }
  auto get_is_const_tensor() const {
    return get().is_const_tensor();
  }
  auto set_const_id(int id) {
    return tmeta_.set_const_id(id);
  }
  auto get_const_id() const {
    return get().get_const_id();
  }
  // set functions can not call const get() to modify
  const habana::TensorExtraMeta& get() const {
    return tmeta_;
  }
  static std::optional<SharedTensorExtraMeta> create(const at::Tensor& tensor) {
    auto impl{tensor.unsafeGetTensorImpl()};
    c10::intrusive_ptr<habana::BaseTensorExtraMeta> meta(
        impl->get_backend_meta_intrusive_ptr());
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
  }
  static std::optional<SharedTensorExtraMeta> create_new(at::Tensor& tensor) {
    auto impl{tensor.unsafeGetTensorImpl()};
    TORCH_CHECK(
        impl != nullptr,
        "Cannot obtain the TensorImpl from the tensor provided");
    c10::intrusive_ptr<habana::BaseTensorExtraMeta> meta(
        impl->get_backend_meta_intrusive_ptr());
    TORCH_CHECK(
        meta == nullptr,
        "Cannot create a new backend meta as one already exists");
    c10::intrusive_ptr<c10::BackendMeta> new_tmeta{
        std::unique_ptr<c10::BackendMeta>(new habana::TensorExtraMeta())};
    impl->set_backend_meta(new_tmeta);
    meta = impl->get_backend_meta_intrusive_ptr();
    TORCH_CHECK(meta == new_tmeta, "Attached meta not the same as created");
    auto tmeta_ptr{dynamic_cast<habana::TensorExtraMeta*>(meta.get())};
    TORCH_CHECK(
        tmeta_ptr != nullptr,
        "Got BackendMeta ",
        meta.get(),
        " but it is not habana::TensorExtraMeta");
    PT_EAGER_DEBUG(
        "Producing SharedTensorExtraMeta for impl : ",
        impl,
        " tensor meta at address ",
        tmeta_ptr,
        " storage address : ",
        tensor.data_ptr());

    return std::optional<SharedTensorExtraMeta>(
        SharedTensorExtraMeta(meta, *tmeta_ptr));
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
      [](const std::string& device_str, bool sync) {
        habana_lazy::HbLazyTensor::StepMarkerBind(device_str, sync);
      },
      py::arg("device_str") = "",
      py::arg("sync") = false);
  m.def("_get_default_generator", []() {
    return habana::detail::getDefaultHPUGenerator();
  });
  py::class_<SharedTensorExtraMeta>(m, "TensorExtraMeta")
      .def_property(
          "is_const_tensor",
          &SharedTensorExtraMeta::get_is_const_tensor,
          &SharedTensorExtraMeta::set_is_const_tensor)
      .def_property(
          "const_id",
          &SharedTensorExtraMeta::get_const_id,
          &SharedTensorExtraMeta::set_const_id);
  m.def("get_tensor_extra_meta", &SharedTensorExtraMeta::create);
  m.def("get_new_tensor_extra_meta", &SharedTensorExtraMeta::create_new);
  m.doc() = "This module registers hpu lazy api.";
}
