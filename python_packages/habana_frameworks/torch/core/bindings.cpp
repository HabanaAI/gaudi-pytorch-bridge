/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "process_group_hcl.h" // "UNUSED" conflict in synapse_helpers/util.h & torch/include/c10d/Types.hpp

#include <pybind11/chrono.h>
#include <torch/extension.h>
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "synapse_helpers/devmem_logger.h"

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;
static void torch_hcl_init() {
  py::object module = py::module::import("torch.distributed");
  py::object register_backend = module.attr("Backend").attr("register_backend");

  register_backend(
      "hcl",
      py::cpp_function(
          &c10d::ProcessGroupHCL::createProcessGroupHCL,
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = std::chrono::milliseconds(40 * 1000)));

  auto processGroup = module.attr("ProcessGroup");
  auto processGroupHCL = intrusive_ptr_class_<::c10d::ProcessGroupHCL>(
      module, "ProcessGroupHCL", processGroup);

  processGroupHCL.def(
      py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                  int rank,
                  int size,
                  std::chrono::milliseconds timeout) {
        return c10::make_intrusive<::c10d::ProcessGroupHCL>(
            store, rank, size, timeout);
      }),
      py::arg("store"),
      py::arg("rank"),
      py::arg("size"),
      py::arg("timeout") = std::chrono::milliseconds(10 * 1000));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch_hcl_init();
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
  m.def("set_dynamic_mode", []() {
    habana_lazy::HbLazyTensor::SetDynamicMode();
  });
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
  m.doc() = "This module registers hpu backend.";
}
