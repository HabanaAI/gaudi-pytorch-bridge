/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <pybind11/chrono.h>
#include <synapse_common_types.h>
#include <torch/extension.h>
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/habana_device/HPUStream.h"
#include "pytorch_helpers/habana_helpers/tensor_info.h"
#include "pytorch_helpers/synapse_helpers/stream.h"

int GetDeviceType() {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return device.type();
}

intptr_t GetDataPtr(const at::Tensor& t) {
  void* data_ptr;
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    data_ptr = habana_lazy::HbLazyTensor::lazyTensorDataPtr(t);
  } else {
    data_ptr = reinterpret_cast<void*>(t.storage().data_ptr().get());
  }

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
  PT_BRIDGE_DEBUG(name, " = min  : ", min, " max : ", max);
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
    void* stream = (void*)d.get_compute_stream(hpu_stream.id());
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
      .export_values();

  m.doc() =
      "This module registers hpu experimental API used by Media internal component.";
}
