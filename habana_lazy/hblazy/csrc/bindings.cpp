/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/csrc/Exceptions.h>

#include <habana_device/hpu_cached_devices.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace hblazy {
namespace {

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() {
    PyEval_RestoreThread(state);
  }
  PyThreadState* state = nullptr;
};

int GetCurrentThreadDevice() {
  auto& d = synapse_helpers::HPURegistrar::get_device();
  return d.id();
}

void StepMarker(int device_id, bool wait) {
  /*
  TODO: Do the equivalent of the following
  XLATensor::SyncLiveTensorsGraph(&device, devices, wait);
  XLATensor::MarkStep(device);
  */
}

void InitModuleBindings(py::module m) {
  m.def("_hb_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def(
      "_hb_step_marker",
      [](int device_id, bool wait) {
        NoGilSection nogil;
        StepMarker(device_id, wait);
      },
      py::arg("device_id"),
      py::arg("wait") = true);
}

} // namespace

void InitBindings(py::module m) {
  InitModuleBindings(m);
}
} // namespace hblazy

PYBIND11_MODULE(_hblazy, m) {
  hblazy::InitBindings(m);
}
