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
#include <synapse_common_types.h>
#include <torch/extension.h>
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/synapse_helpers/stream.h"

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

bool IsAvailable() {
  try {
    habana::HABANAGuardImpl device_guard;
    device_guard.getDevice();
    auto& device = synapse_helpers::HPURegistrar::get_device();
    if (device.get_count_by_current_type() > 0)
      return true;
  } catch (...) {
    return false;
  }
  return false;
}

int GetDeviceType() {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return device.type();
}

int GetCurrentThreadDevice() {
  auto& d = synapse_helpers::HPURegistrar::get_device();
  return d.id();
}

intptr_t GetDataPtr(const at::Tensor& t) {
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0,
      "htcore.data_ptr() is only available for lazy mode."
      " Set PT_HPU_LAZY_MODE=1 or PT_HPU_LAZY_MODE=2.");
  return reinterpret_cast<intptr_t>(
      habana_lazy::HbLazyTensor::lazyTensorDataPtr(t));
}

const std::string get_device_name(int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return device.name();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch_hcl_init();
  // python API to report device memory live allocation details
  m.def("memstat_livealloc", [](const char* msg = "") {
    habana::HPUDeviceAllocator::print_memory_stats(msg);
  });
  m.def("get_fallback_op_count", []() {
    return habana::HpuFallbackHelper::get()->get_op_count();
  });
  m.def("_hb_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def("current_device", []() { return GetCurrentThreadDevice(); });
  m.def("is_available", []() { return IsAvailable(); });
  m.def("get_device_type", []() { return GetDeviceType(); });
  m.def("is_enabled_weight_permute_pass", []() {
    return habana_lazy::exec::OptPassCfg::GetInstance()
        ->IsEnabledWeightPermutePass();
  });
  m.def("synchronize_device", []() {
    synapse_helpers::HPURegistrar::synchronize_device();
  });
  m.def("get_device_count", []() {
    return synapse_helpers::HPURegistrar::get_total_device_count();
  });
  m.def("get_device_name", [](int id) { return get_device_name(id); });

  // Lazy apis
  m.def(
      "_mark_step",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::StepMarkerBind(device_str);
      },
      py::arg("device_str") = "");
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
  m.def(
      "enable_weight_permute_pass",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetWeightPermutePass(
            flag);
      },
      py::arg("flag"));
  m.def(
      "enable_replace_views",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetReplaceViews(flag);
      },
      py::arg("flag"));
  m.def(
      "set_module_name",
      [](const std::string& name) {
        habana_lazy::ir::setCurrentModuleName(name);
      },
      py::arg("name"));
  m.def(
      "data_ptr",
      [](const at::Tensor& t) { return GetDataPtr(t); },
      py::arg("t"));
  m.def("compute_stream", []() {
    if (IsAvailable()) {
      auto& d = synapse_helpers::HPURegistrar::get_device();
      void* stream = (void*)d.get_compute_stream();
      return reinterpret_cast<intptr_t>(stream);
    }
    return reinterpret_cast<intptr_t>(nullptr);
  });
  py::enum_<synDeviceType>(m, "synDeviceType")
      .value("synDeviceGaudi", synDeviceGaudi)
      .value("synDeviceGaudi2", synDeviceGaudi2)
      .export_values();

  m.doc() = "This module registers hpu backend.";
}
