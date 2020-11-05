/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <c10/core/Device.h>
#include <torch/csrc/Exceptions.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <habana_device/hpu_cached_devices.h>
#include <habana_lazy/hlexec.h>
#include <habana_lazy/hpu_lazy_tensors.h>

namespace py = pybind11;

namespace habana_lazy {
namespace {

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() {
    PyEval_RestoreThread(state);
  }
  PyThreadState* state = nullptr;
};

c10::Device SynapseDeviceToAtenDevice(const synapse_helpers::device& device) {
  return c10::Device(at::kHABANA, device.id());
}

const synapse_helpers::device& AtenDeviceToSynapseDevice(
    const c10::Device& device) {
  TORCH_CHECK(device.type() == at::kHABANA);
  const int index = device.has_index() ? device.index() : 0;
  return synapse_helpers::HPURegistrar::get_device(index);
}

c10::Device GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return SynapseDeviceToAtenDevice(
        synapse_helpers::HPURegistrar::get_device());
  }

  return c10::Device(device_str);
}

std::string GetCurrentThreadDevice() {
  return SynapseDeviceToAtenDevice(synapse_helpers::HPURegistrar::get_device())
      .str();
}

void StepMarker(
    const std::string& device_str,
    const std::vector<std::string>& devices) {
  c10::Device device = GetDeviceOrCurrent(device_str);
  HbLazyTensor::SyncLiveTensorsGraph(&device, devices);
  HbLazyTensor::MarkStep(device);
}

void InitModuleBindings(py::module m) {
  m.def("_hb_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def(
      "_hb_step_marker",
      [](const std::string& device_str,
         const std::vector<std::string>& devices) {
        NoGilSection nogil;
        StepMarker(device_str, devices);
      },
      py::arg("device_str"),
      py::arg("devices"));
  m.def(
      "_enable_eliminate_common_subexpression",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->enable_eliminate_common_subexpression =
            flag;
      },
      py::arg("flag"));
  m.def(
      "_enable_eliminate_dead_code",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->enable_eliminate_dead_code =
            flag;
      },
      py::arg("flag"));
  m.def(
      "_enable_constant_pooling",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->enable_constant_pooling =
            flag;
      },
      py::arg("flag"));
}

} // namespace

void InitBindings(py::module m) {
  InitModuleBindings(m);
}
} // namespace habana_lazy

PYBIND11_MODULE(_hblazy, m) {
  habana_lazy::InitBindings(m);
}
