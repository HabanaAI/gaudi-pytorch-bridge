/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/record_function.h>
#include <habana_lazy/hlexec.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <functional>
#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_bridge/passes/habana_fuser.h"
#include "habana_bridge/passes/remove_inplace_ops.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "synapse_helpers/devmem_logger.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // python API to report device memory live allocation details
  m.def("memstat_livealloc", [](const char* msg = "") {
    synapse_helpers::print_live_allocations(msg);
  });

  // Lazy apis
  m.def(
      "mark_step",
      [](const std::string& device_str, bool is_blocking) {
        if (is_blocking) {
          habana_lazy::HbLazyTensor::StepMarkerBlocking(device_str);
        } else {
          habana_lazy::HbLazyTensor::StepMarker(device_str);
        }
      },
      py::arg("device_str") = "",
      py::arg("is_blocking") = false);
  m.def(
      "run_saved_model",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::RunSavedGraph(device_str);
      },
      py::arg("device_str") = "");
  m.def(
      "enable_eliminate_common_subexpression",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetCSEElimination(flag);
      },
      py::arg("flag"));
  m.def(
      "enable_eliminate_dead_code",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetDeadCodeElimination(
            flag);
      },
      py::arg("flag"));
  m.def(
      "enable_constant_pooling",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetConstPooling(flag);
      },
      py::arg("flag"));
  m.def(
      "enable_peephole_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetPeepholeOpt(flag);
      },
      py::arg("flag"));
  m.def(
      "enable_fuse_t_mm_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetFuseTMM(flag);
      },
      py::arg("flag"));
  m.def(
      "enable_fuse_bn_relu_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetFuseBnRelu(flag);
      },
      py::arg("flag"));
  m.def(
      "enable_permute_pass",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetPermutePass(flag);
      },
      py::arg("flag"));
  m.def(
      "enable_replace_inplace_ops",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetReplaceInplaceOps(
            flag);
      },
      py::arg("flag"));
  m.doc() = "This module registers habana backend.";
}
