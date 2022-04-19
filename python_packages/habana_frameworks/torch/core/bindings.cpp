/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <pybind11/chrono.h>
#include <torch/extension.h>
#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"

int GetCurrentThreadDevice() {
  auto& d = synapse_helpers::HPURegistrar::get_device();
  return d.id();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_hb_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def(
      "_mark_step",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::StepMarkerBind(device_str);
      },
      py::arg("device_str") = "");

  // FIME removed in the next patch were debug is added and
  // it will be moved to debug
  m.def("get_fallback_op_count", []() {
    return habana::HpuFallbackHelper::get()->get_op_count();
  });
  m.def(
      "enable_eliminate_common_subexpression",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetCSEElimination(flag);
      },
      py::arg("flag"));
  m.def(
      "enable_weight_permute_pass",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetWeightPermutePass(
            flag);
      },
      py::arg("flag"));
  m.def("is_enabled_weight_permute_pass", []() {
    return habana_lazy::exec::OptPassCfg::GetInstance()
        ->IsEnabledWeightPermutePass();
  });
  m.def(
      "enable_constant_pooling",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetConstPooling(flag);
      },
      py::arg("flag"));

  m.def("memstat_livealloc", [](const char* msg = "") {
    habana::HPUDeviceAllocator::print_memory_stats(msg);
  });

  m.def(
      "memstat_devmem_start_collect",
      [](const char* msg = "", bool show_leaked_callstacks = true) {
        habana::HPUDeviceAllocator::memstat_devmem_start_collect(
            msg, show_leaked_callstacks);
      });
  m.def("memstat_devmem_stop_collect", [](const char* msg = "") {
    habana::HPUDeviceAllocator::memstat_devmem_stop_collect(msg);
  });

  m.def("is_enabled_synapse_layout_handling", []() {
    return GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING);
  });
  m.def(
      "set_module_name",
      [](const std::string& name) {
        habana_lazy::ir::setCurrentModuleName(name);
      },
      py::arg("name"));

  m.doc() = "This module registers hpu lazy api.";
}
