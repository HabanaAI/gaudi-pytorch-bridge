/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>

#include <habana_lazy/hlexec.h>
#include "habana_lazy/hpu_lazy_tensors.h"
#include "register.h"

namespace py = pybind11;

static struct ::habana::hb_torch_opts opts;

PYBIND11_MODULE(hb_torch, m) {
  std::function<::habana::hb_torch_opts()> get_options = []() { return opts; };
  habana::torch_habana_register_fusion_pass(get_options);
  habana::torch_habana_register_pre_diff_pass(get_options);
  // python API to enable and disable tvm fusion
  m.def("enable", []() { opts.fusion_enabled = true; });
  m.def("disable", []() { opts.fusion_enabled = false; });
  m.def("remove_inplace_ops", []() { opts.remove_inplace_ops = true; });

  // python API to report device memory live allocation details
  m.def("memstat_livealloc", [](const char* msg = "") {
    synapse_helpers::print_live_allocations(msg);
  });

  // Lazy apis
  m.def(
      "mark_step",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::StepMarker(device_str);
      },
      py::arg("device_str") = "");
  m.def(
      "run_saved_model",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::RunSavedGraph(device_str);
      },
      py::arg("device_str") = "");
  m.def(
      "enable_eliminate_common_subexpression",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_eliminate_common_subexpression = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_eliminate_dead_code",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_eliminate_dead_code = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_constant_pooling",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->enable_constant_pooling =
            flag;
      },
      py::arg("flag"));
  m.def(
      "enable_peephole_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_peephole_optimization = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_fuse_t_mm_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_fuse_t_mm_optimization = flag;
      },
      py::arg("flag"));
  m.def(
      "enable_fuse_bn_relu_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()
            ->enable_fuse_bn_relu_optimization = flag;
      },
      py::arg("flag"));
  m.doc() = "This module registers habana backend.";
}
