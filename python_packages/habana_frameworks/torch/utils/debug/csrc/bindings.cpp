/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <synapse_api.h>
#include <torch/extension.h>
#include <map>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_eager/eager_context.h"
#include "habana_helpers/towl.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_kernels/shape_agnostic_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/memlog.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/misc_utils.h"

enum log_level {
  TRACE = HLLOG_LEVEL_TRACE,
  DEBUG = HLLOG_LEVEL_DEBUG,
  INFO = HLLOG_LEVEL_INFO,
  WARN = HLLOG_LEVEL_WARN,
  ERROR = HLLOG_LEVEL_ERROR,
  CRITICAL = HLLOG_LEVEL_CRITICAL
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_fallback_op_count", []() {
    return habana::HpuFallbackHelper::get()->get_op_count();
  });
  m.def("get_shape_agnostic_unsupported_ops", []() {
    habana::TryJoinPendingEagerPipelineThreads();
    return habana::HpuShapeAgnosticHelper::get()
        ->get_shape_agnostic_unsupported_ops();
  });
  m.def("get_eager_compiler_unsupported_op_prefixes", []() {
    habana::TryJoinPendingEagerPipelineThreads();
    return habana::HpuShapeAgnosticHelper::get()
        ->get_eager_compiler_unsupported_op_prefixes();
  });
  m.def("get_jit_cache_size", []() {
    habana::TryJoinPendingEagerPipelineThreads();
    return habana::HpuShapeAgnosticHelper::get()->get_jit_cache_size();
  });
  m.def("clear_jit_cache", []() {
    habana::TryJoinPendingEagerPipelineThreads();
    return habana::HpuShapeAgnosticHelper::get()->clear_jit_cache();
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
      "enable_fuse_conv_bn_optimization",
      [](const bool flag) {
        habana_lazy::exec::OptPassCfg::GetInstance()->SetFuseConvBn(flag);
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
    habana_lazy::exec::HlExec::LoadDSCheckpoint(path);
  });
  m.def("save_ds_checkpoint", [](std::string path) {
    habana_lazy::exec::HlExec::SaveDSCheckpoint(path);
  });
  m.def("clear_dynamic_bucket_recipe_info", []() {
    habana::ClearDynamicBucketRecipeInfo();
  });
  m.def("is_enabled_lazy_collectives", []() {
    return GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_COLLECTIVES);
  });
  m.def("hb_print", [](const char* msg) { PT_CUSTOM_DEBUG(msg); });
  m.def("towl_print", [](const std::string& msg) {
    towl::emitPythonString(msg);
  });
  m.def("towl_configure", [](bool flag, std::string config) {
    towl::configure(flag, config);
  });
  m.doc() = "This module registers hpu host debug API";
  m.def(
      "mem_log", [](std::string msg) { habana_lazy::log_dev_mem_stats(msg); });
  m.def("dump_memory_reporter", []() {
    return habana::HPUDeviceAllocator::dump_memory_reporter();
  });
  m.def("_disk_cache_flush", []() {
    habana::HPUDeviceContext::flush_disk_cache();
  });
  m.def("hg_print", [](std::string msg) { PT_HPUGRAPH_DEBUG(msg); });
  py::enum_<log_level>(m, "log_level")
      .value("trace", TRACE)
      .value("debug", DEBUG)
      .value("info", INFO)
      .value("warn", WARN)
      .value("error", ERROR)
      .value("critical", CRITICAL)
      .export_values();
  m.def("is_log_python_enabled", [](log_level level) {
    return hl_logger::logLevelAtLeast(HlLogger::LoggerType::PT_PYTHON, level);
  });
  m.def("log_python", [](log_level level, const std::string& message) {
    switch (level) {
      case HLLOG_LEVEL_TRACE:
        PT_PYTHON_TRACE(message);
        break;
      case HLLOG_LEVEL_DEBUG:
        PT_PYTHON_DEBUG(message);
        break;
      case HLLOG_LEVEL_INFO:
        PT_PYTHON_INFO(message);
        break;
      case HLLOG_LEVEL_WARN:
        PT_PYTHON_WARN(message);
        break;
      case HLLOG_LEVEL_ERROR:
        PT_PYTHON_ERROR(message);
        break;
      case HLLOG_LEVEL_CRITICAL:
        PT_PYTHON_CRITICAL(message);
        break;
      default:
        PT_PYTHON_FATAL("Received an unknown log level: ", level);
        break;
    }
  });
  m.def(
      "enable_logging",
      [](const std::string& logger_name, log_level logger_level) {
        hl_logger::setLoggingLevelByMask(logger_name, logger_level);
      });
  m.def("get_pt_logging_levels", []() {
    std::map<std::string, int> result;
    for (int i = 0; i < static_cast<int>(HlLogger::LoggerType::LOG_MAX); i++) {
      HlLogger::LoggerType logger = static_cast<HlLogger::LoggerType>(i);
      result.insert(std::pair<std::string, int>(
          hl_logger::getLoggerEnumItemName(logger),
          hl_logger::getLoggingLevel(logger)));
    }
    return result;
  });
  m.def("dump_state_and_terminate", [](const char* msg, uint64_t flags) {
    synDumpStateAndTerminate(msg, flags);
  });
}
