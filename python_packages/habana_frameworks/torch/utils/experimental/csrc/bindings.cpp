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
#include <ATen/Tensor.h>
#include <synapse_common_types.h>
#include <torch/extension.h>
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/tensor_info.h"
#include "common/utils.h"

int GetDeviceType() {
  auto& device = habana::HPURegistrar::get_device();
  return device.type();
}

intptr_t GetDataPtr(const at::Tensor& t) {
  void* data_ptr = common::GetDataPtrFromTensor(t);

  if (data_ptr) {
    size_t device_id = t.device().index();
    auto& device = habana::HPURegistrar::get_device(device_id).syn_device();

    auto address = reinterpret_cast<void*>(device.get_fixed_address(data_ptr));
    return reinterpret_cast<intptr_t>(address);
  }

  return 0;
}

void SetProfilerTracerMemory(const int device_id) {
  if (device_id >= 0) {
    uint32_t bytes_req = 0;
    synStatus status = synProfilerQueryRequiredMemory(0, &bytes_req);
    if (status != synSuccess) {
      std::cerr << "synProfilerQueryRequiredMemory failed" << std::endl;
    }

    if (bytes_req > 0) {
      void* data_ptr{nullptr};
      auto& device = habana::HPURegistrar::get_device(device_id);
      device.get_device_memory().malloc(&data_ptr, bytes_req);
      auto user_buff = reinterpret_cast<void*>(
          device.syn_device().get_fixed_address(data_ptr));
      status = synProfilerSetUserBuffer(0, user_buff);
      if (status != synSuccess) {
        std::cerr << "synProfilerSetUserBuffer failed" << std::endl;
      }
    }
  }
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
  auto& device = habana::HPURegistrar::get_device().syn_device();
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
    auto& d = habana::HPURegistrar::get_device().syn_device();
    auto hpu_stream = c10::hpu::getDefaultHPUStream(d.id());
    void* stream = (void*)d.get_stream(hpu_stream.id());
    return reinterpret_cast<uintptr_t>(stream);
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
  m.def(
      "set_profiler_tracer_memory",
      [](const int device_id) { return SetProfilerTracerMemory(device_id); },
      py::arg("device_id"));
  py::enum_<synDeviceType>(m, "synDeviceType")
      .value("synDeviceGaudi", synDeviceGaudi)
      .value("synDeviceGaudi2", synDeviceGaudi2)
      .value("synDeviceGaudi3", synDeviceGaudi3)
      .export_values();

  m.doc() =
      "This module registers hpu experimental API used by Media internal component.";
}
