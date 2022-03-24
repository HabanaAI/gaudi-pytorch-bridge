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
#include <synapse_common_types.h>
#include <torch/extension.h>
#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/habana_helpers/dynamic_bucket_info.h"
#include "pytorch_helpers/synapse_helpers/stream.h"

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
  // python APIs to report device memory live allocation details
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

  // python APIs related to dynamic shape bucket refinement
  m.def("dump_refined_recipe_stat", []() {
    habana_helpers::DynamicBucketInfo::DumpDynamicRecipeStat();
  });
  m.def("disable_bucket_refinement", []() {
    habana_helpers::DynamicBucketInfo::DisableBucketRefinement();
  });
  m.def("dump_bucket_memory_stat", []() {
    habana::DynamicBucketInfoMap::DumpBucketMemoryStat();
  });
  m.def("dump_history_memory_stat", []() {
    habana::DynamicBucketInfoMap::DumpHistoryMemoryStat();
  });
  m.def("dump_recipe_memory_stat", []() {
    habana::RecipeCacheLRU::DumpRecipeMemoryStat();
  });
  m.def("dump_synapse_recipe_memory_stat", []() {
    habana::RecipeCacheLRU::DumpSynapseRecipeMemoryStat();
  });
  m.def("dump_dynamic_shape_memory_stat", []() {
    habana::RecipeCacheLRU::DumpDynamicShapeMemoryStat();
  });

  // python APIs related to cpu fallback
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

  // python APIs related to lazy mode
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
      .value("synDeviceGaudiM", synDeviceGaudiM)
      .value("synDeviceGaudi2", synDeviceGaudi2)
      .export_values();

  m.doc() = "This module registers hpu backend.";
}
