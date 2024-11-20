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
#include <ATen/Tensor.h>
#include <synapse_common_types.h>
#include <torch/extension.h>
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/event_dispatcher.h"
#include "backend/helpers/tensor_info.h"
#include "backend/synapse_helpers/device.h"
#include "common/utils.h"

intptr_t GetDataPtr(const at::Tensor& t) {
  void* data_ptr = common::GetDataPtrFromTensor(t);

  if (data_ptr) {
    size_t device_id = t.device().index();
    auto& device = habana::HPUDeviceContext::get_device(device_id);

    auto address = reinterpret_cast<void*>(device.get_fixed_address(data_ptr));
    return reinterpret_cast<intptr_t>(address);
  }

  return 0;
}

void SetProfilerTracerMemory(const uint32_t device_id) {
  uint32_t bytes_req = 0;
  synStatus status = synProfilerQueryRequiredMemory(device_id, &bytes_req);
  if (status != synSuccess) {
    std::cerr << "synProfilerQueryRequiredMemory failed" << std::endl;
  }

  if (bytes_req > 0) {
    void* data_ptr{nullptr};
    auto& device = habana::HPUDeviceContext::get_device(device_id);
    device.get_device_memory().malloc(&data_ptr, bytes_req);
    auto user_buff =
        reinterpret_cast<void*>(device.get_fixed_address(data_ptr));
    status = synProfilerSetUserBuffer(device_id, user_buff);
    if (status != synSuccess) {
      std::cerr << "synProfilerSetUserBuffer failed" << std::endl;
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
  auto& device = habana::HPUDeviceContext::get_device();
  device.record_param(
      name, is_param, is_grad, is_optim_state, t_start, t_start + t_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_device_type", []() {
    return synapse_helpers::device::get_device_type();
  });
  m.def(
      "data_ptr",
      [](const at::Tensor& t) { return GetDataPtr(t); },
      py::arg("t"));
  m.def("compute_stream", []() {
    auto& d = habana::HPUDeviceContext::get_device();
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
      [](const uint32_t device_id) {
        return SetProfilerTracerMemory(device_id);
      },
      py::arg("device_id"));
  py::enum_<synDeviceType>(m, "synDeviceType")
      .value("synDeviceGaudi", synDeviceGaudi)
      .value("synDeviceGaudi2", synDeviceGaudi2)
      .value("synDeviceGaudi3", synDeviceGaudi3)
      .export_values();

  m.def("reset_device_memory", []() {
    auto& device = habana::HPUDeviceContext::get_device();
    device.cleanup_workspace_buffer();
    device.get_device_memory().reset_pool();
    habana_helpers::EventDispatcher::Instance().unsubscribe_all();
  });
  m.doc() =
      "This module registers hpu experimental API used by Media internal component.";
}
