/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include <synapse_common_types.h>
#include <torch/extension.h>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/tensor_info.h"
#include "backend/synapse_helpers/stream.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"

int GetDeviceType() {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return device.type();
}

intptr_t GetDataPtr(const at::Tensor& t) {
  void* data_ptr;
  data_ptr = habana_lazy::HbLazyTensor::lazyTensorDataPtr(t);

  if (data_ptr) {
    size_t device_id = t.device().index();
    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

    auto address = reinterpret_cast<void*>(device.get_fixed_address(data_ptr));
    return reinterpret_cast<intptr_t>(address);
  }

  return 0;
}

void RecordQuantParams(std::string name, float min, float max) {
  PtTensorInferenceData::get_instance().SetInferenceTensorRange(name, min, max);
  std::replace(name.begin(), name.end(), '.', '/'); // replace all 'x' to 'y'
  PT_BRIDGE_DEBUG("Quantization Record", " ", name, " ", min, " ", max);
}

void RecordParam(
    const std::string& name,
    const bool is_param,
    const bool is_grad,
    const bool is_optim_state,
    const uint64_t t_start,
    const uint64_t t_size) {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  device.record_param(
      name, is_param, is_grad, is_optim_state, t_start, t_start + t_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_device_type", []() { return GetDeviceType(); });
  m.def(
      "data_ptr",
      [](const at::Tensor& t) { return GetDataPtr(t); },
      py::arg("t"));
  m.def("compute_stream", []() {
    auto& d = synapse_helpers::HPURegistrar::get_device();
    HPUStream hpu_stream = getDefaultHPUStream(d.id());
    void* stream = (void*)d.get_stream(hpu_stream.id());
    return reinterpret_cast<intptr_t>(stream);
  });
  m.def(
      "record_quant_param",
      [](std::string name, float min, float max) {
        RecordQuantParams(name, min, max);
      },
      py::arg("name"),
      py::arg("min"),
      py::arg("max"));
  m.def(
      "record_param",
      [](const std::string name,
         const bool is_param,
         const bool is_grad,
         const bool is_optim_state,
         const uint64_t t_start,
         const uint64_t t_size) {
        RecordParam(name, is_param, is_grad, is_optim_state, t_start, t_size);
      },
      py::arg("name"),
      py::arg("is_param"),
      py::arg("is_grad"),
      py::arg("is_optim_state"),
      py::arg("t_start"),
      py::arg("t_size"));
  py::enum_<synDeviceType>(m, "synDeviceType")
      .value("synDeviceGaudi", synDeviceGaudi)
      .value("synDeviceGaudi2", synDeviceGaudi2)
      .value("synDeviceGreco", synDeviceGreco)
      .value("synDeviceGaudi3", synDeviceGaudi3)
      .export_values();

  m.doc() =
      "This module registers hpu experimental API used by Media internal component.";
}
