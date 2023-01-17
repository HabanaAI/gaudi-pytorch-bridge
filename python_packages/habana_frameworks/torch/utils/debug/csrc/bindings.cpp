/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/extension.h>
#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/memlog.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_fallback_op_count", []() {
    return habana::HpuFallbackHelper::get()->get_op_count();
  });
  m.def("is_enabled_weight_permute_pass", []() {
    return habana_lazy::exec::OptPassCfg::GetInstance()
        ->IsEnabledWeightPermutePass();
  });
  m.def("set_dynamic_mode", []() {
    habana_lazy::HbLazyTensor::SetDynamicMode();
  });

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
      "enable_bn_param_recalculation",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetBnParamRecalc(flag);
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
  m.def("load_ds_checkpoint", [](std::string path) {
    habana::DynamicBucketInfoMap::load_ds_checkpoint(path.c_str());
  });
  m.def("save_ds_checkpoint", [](std::string path) {
    habana::DynamicBucketInfoMap::save_ds_checkpoint(path.c_str());
  });
  m.def("is_enabled_synapse_layout_handling", []() {
    return GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING);
  });
  m.def("clear_dynamic_bucket_recipe_info", []() {
    habana::ClearDynamicBucketRecipeInfo();
  });
  m.def("is_enabled_lazy_collectives", []() {
    return GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_COLLECTIVES);
  });
  m.def("hb_print", [](const char* msg) { PT_CUSTOM_DEBUG(msg); });
  m.doc() = "This module registers hpu host debug API";
  m.def(
      "mem_log", [](std::string msg) { habana_lazy::log_dev_mem_stats(msg); });
  m.def("bridge_cleanup", []() { habana::HabanaLaunchOpPT::cleanUp(); });
  m.def("dump_memory_reporter", []() {
    return habana::HPUDeviceAllocator::dump_memory_reporter();
  });
}
