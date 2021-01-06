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

void InitModuleBindings(py::module m) {
  m.def("_hb_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def(
      "_hb_step_marker",
      [](const std::string& device_str) {
        NoGilSection nogil;
        HbLazyTensor::StepMarker(device_str);
      },
      py::arg("device_str"));
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
  m.def(
      "_enable_peephole_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->enable_peephole_optimization =
            flag;
      },
      py::arg("flag"));
  m.def(
      "_enable_fuse_t_mm_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_fuse_t_mm_optimization = flag;
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
