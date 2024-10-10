/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "backend/scalar_cache.h"
#include "habana_lazy/memlog.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

//clang-format off
#include <ATen/autocast_mode.h>
#include <pybind11/chrono.h>
#include <synapse_common_types.h>
#include <torch/extension.h>
//clang-format on
#include <tuple>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/HPUGraph.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/stream.h"
#include "habana_lazy/tensor_impl.h"
#include "habana_lazy/view_utils.h"
#include "hpu_ops/custom_op_outshape.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"

#include "python_packages/habana_frameworks/torch/hpu/csrc/Event.h"
#include "python_packages/habana_frameworks/torch/hpu/csrc/Module.h"
#include "python_packages/habana_frameworks/torch/hpu/csrc/Stream.h"

using namespace c10::hpu;

void hpu_init() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  device.get_count_by_current_type();
  // later will add device properties here.
}

const std::string get_device_name([[maybe_unused]] int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  return habana::HPURegistrar::get_device().name();
}

/* clang-format off */
const synapse_helpers::MemoryStats get_mem_stat(
    [[maybe_unused]] int device_id) {
  /* clang-format on */
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = habana::HPURegistrar::get_device();
  synapse_helpers::MemoryStats stats;
  device.get_device_memory().get_memory_stats(&stats);
  return stats;
}

const std::string get_hlml_shared_object_name([[maybe_unused]] int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = habana::HPURegistrar::get_device();
  auto& device_memory = device.get_device_memory();
  auto hlml_reporter = device_memory.get_hlml_memory_reporter();
  if (hlml_reporter) {
    return hlml_reporter->GetPath();
  }
  return "";
}

void reset_peak_memory_stats([[maybe_unused]] int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = habana::HPURegistrar::get_device();
  device.get_device_memory().reset_peak_memory_stats();
}

void clear_memory_stats([[maybe_unused]] int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = habana::HPURegistrar::get_device();
  device.get_device_memory().clear_memory_stats();
}

const std::string get_mem_stat_summary(int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto stats = get_mem_stat(device_id);
  // return only memory info skip poll id, mask info etc..
  std::string summary = absl::StrFormat(
      "  Limit:             %20lld (%.2f GB)\n"
      "  InUse:             %20lld (%.2f MB)\n"
      "  MaxInUse:          %20lld (%.2f MB)\n"
      "  NumAllocs:         %20lld\n"
      "  NumFrees:          %20lld\n"
      "  MaxAllocSize:      %20lld (%.2f MB)\n"
      "  ActiveAllocs:      %20lld\n"
      "%s\n",
      stats.memory_limit,
      stats.memory_limit / (1024 * 1024 * 1024.),
      stats.bytes_in_use,
      stats.bytes_in_use / (1024 * 1024.),
      stats.peak_bytes_in_use,
      stats.peak_bytes_in_use / (1024 * 1024.),
      stats.num_allocs,
      stats.num_frees,
      stats.largest_alloc_size,
      stats.largest_alloc_size / (1024 * 1024.),
      (int64_t)stats.num_allocs - (int64_t)stats.num_frees,
      "");
  return summary;
}

const py::dict get_extended_mem_stat_summary() {
  using namespace pybind11::literals;
  auto stats = get_mem_stat(0);
  auto& device = habana::HPURegistrar::get_device();

  auto persistent =
      (int64_t)stats.bytes_in_use - (int64_t)stats.scratch_mem_in_use;
  auto max_cntgs_chunk = device.get_device_memory().get_max_cntgs_chunk_size();
  auto future_bytes = habana_lazy::get_future_memory().first;

  return py::dict(
      "limit"_a = stats.memory_limit,
      "in_use"_a = stats.bytes_in_use,
      "persistent"_a = persistent,
      "workspace"_a = stats.scratch_mem_in_use,
      "last_workspace"_a = device.syn_device().get_real_workspace_size(),
      "future"_a = future_bytes,
      "max_cntgs_chunk"_a = max_cntgs_chunk,
      "max_in_use"_a = stats.peak_bytes_in_use,
      "num_allocs"_a = stats.num_allocs,
      "num_free"_a = stats.num_frees,
      "max_alloc_size"_a = stats.largest_alloc_size,
      "active_allocs"_a = (int64_t)stats.num_allocs - (int64_t)stats.num_frees);
}

// In parallel accumulation, it is possible to have a case when program
// is finishing and some workload is still pending in accumulation/cleanup
// threads (i.e. user called ops, but never requested the output values).
// To avoid race between Python shuting down and accumulation threads finishing
// work, it's recommended to sync those threads in Python 'atexit' registry.
void sync_threads() {
  auto gil_release = pybind11::gil_scoped_release();
  habana_lazy::AccThread::Get().SyncAccThreadPool();
  habana_lazy::AccThread::Get().ExecuteAllCleanupTasks();
}

void clear_global_context() {
  auto& d = habana::HPURegistrar::get_device().get_scalar_cache();
  d.ClearCache();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto module = m.ptr();
  THP_HPU_Stream_init(module);
  THP_HPU_Event_init(module);
  PyModule_AddFunctions(module, THP_HPU_Module_methods());

  m.def("init", []() { hpu_init(); });
  m.def("cleanup", []() {
    sync_threads();
    clear_global_context();
  });
  m.def("current_device", []() {
    auto& d = habana::HPURegistrar::get_device();
    return d.id();
  });
  m.def("synchronize_device", []() {
    // Need to finish execution all the performed operations till now and has to
    // include also the accumulated lazy graph ops and Then wait for device
    // sync.
    // Note: This is synchronous step marker
    PT_IRGRAPH_DEBUG("step marker due to bindings-synchronize_device");
    habana_lazy::HbLazyTensor::StepMarker();
    PT_IRGRAPH_DEBUG(
        "synchronize host multistage pipeline due to bindings-synchronize_device");
    habana::HPURegistrar::synchronize_host_multistage_pipeline();
    habana::HPURegistrar::synchronize_device();
  });
  m.def("device_count", []() {
    return habana::HPURegistrar::get_total_device_count();
  });
  m.def("get_device_capability", []() {
    return habana::HPURegistrar::get_device_capability();
  });
  m.def("get_device_properties", [](unsigned id) {
    return habana::HPURegistrar::get_device_properties(id);
  });
  m.def("reset_peak_memory_stats", [](int id) { reset_peak_memory_stats(id); });
  m.def("clear_memory_stats", [](int id) { clear_memory_stats(id); });
  m.def("get_mem_stats", [](int id) {
    using namespace pybind11::literals;
    auto stats = get_mem_stat(id);
    py::dict d(
        "Limit"_a = stats.memory_limit,
        "InUse"_a = stats.bytes_in_use,
        "MaxInUse"_a = stats.peak_bytes_in_use,
        "NumAllocs"_a = stats.num_allocs,
        "NumFrees"_a = stats.num_frees,
        "ActiveAllocs"_a =
            ((int64_t)stats.num_allocs - (int64_t)stats.num_frees),
        "MaxAllocSize"_a = stats.largest_alloc_size,
        "TotalSystemAllocs"_a = stats.total_allocs,
        "TotalSystemFrees"_a = stats.total_frees,
        "TotalActiveAllocs"_a =
            ((int64_t)stats.total_allocs - (int64_t)stats.total_frees));
    return d;
  });
  m.def("get_memory_summary", [](int id) {
    auto mem_stat_str = get_mem_stat_summary(id);
    return mem_stat_str;
  });
  m.def("get_hlml_shared_object_name", [](int id) {
    return get_hlml_shared_object_name(id);
  });
  m.def("get_extended_memory_summary", []() {
    return get_extended_mem_stat_summary();
  });
  m.def("setDeterministic", [](bool val) {
    auto& gconfig = habana::HPURegistrar::get_hpu_global_config();
    gconfig.setDeterministic(val);
  });
  m.def("getDeterministic", []() -> bool {
    return habana::HPURegistrar::get_hpu_global_config().getDeterministic();
  });
  m.def("get_device_name", [](int id) { return get_device_name(id); });
#if IS_PYTORCH_AT_LEAST(2, 4)
  m.def("set_autocast_hpu_enabled", [](py::object enabled) {
    at::autocast::set_autocast_enabled(at::kHPU, enabled.ptr() == Py_True);
  });
  m.def("is_autocast_hpu_enabled", []() {
    return at::autocast::is_autocast_enabled(at::kHPU);
  });
  m.def("set_autocast_hpu_dtype", [](py::object dtype) {
    at::ScalarType targetType =
        reinterpret_cast<THPDtype*>(dtype.ptr())->scalar_type;
    at::autocast::set_autocast_dtype(at::kHPU, targetType);
  });
  m.def("get_autocast_hpu_dtype", []() {
    at::ScalarType current_dtype = at::autocast::get_autocast_dtype(at::kHPU);
    auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
    return py::reinterpret_borrow<py::object>(dtype);
  });
#else
  m.def("set_autocast_hpu_enabled", [](py::object enabled) {
    at::autocast::set_hpu_enabled(enabled.ptr() == Py_True);
  });
  m.def("is_autocast_hpu_enabled", []() {
    return at::autocast::is_hpu_enabled();
  });
  m.def("set_autocast_hpu_dtype", [](py::object dtype) {
    at::ScalarType targetType =
        reinterpret_cast<THPDtype*>(dtype.ptr())->scalar_type;
    at::autocast::set_autocast_hpu_dtype(targetType);
  });
  m.def("get_autocast_hpu_dtype", []() {
    at::ScalarType current_dtype = at::autocast::get_autocast_hpu_dtype();
    auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
    return py::reinterpret_borrow<py::object>(dtype);
  });
#endif
  m.def("get_view_hash", [](at::Tensor t) -> size_t {
    size_t hash = 0;
    auto hl_t = habana_lazy::TryGetHbLazyTensor(t);
    c10::optional<habana_lazy::HbLazyTensor> hl_view = c10::nullopt;
    while (hl_t && hl_t->getDataPtr()->stride_params.has_value()) {
      auto& params = hl_t->getDataPtr()->stride_params.value();
      if (params.optype != habana_lazy::StridedOPType::kStridedOpView) {
        if (hl_view.has_value()) {
          hash = habana_lazy::HbLazyTensorViews::updateViewHash(
              *hl_view, (size_t)hash);
          hl_view = c10::nullopt;
        }
        hash =
            habana_lazy::HbLazyTensorViews::updateViewHash(*hl_t, (size_t)hash);
      } else {
        hl_view = hl_t;
      }
      t = (params.optype == habana_lazy::StridedOPType::kStridedOpDefault)
          ? params.base
          : params.parent;
      hl_t = habana_lazy::TryGetHbLazyTensor(t);
    }
    if (hl_view.has_value()) {
      hash = habana_lazy::HbLazyTensorViews::updateViewHash(
          *hl_view, (size_t)hash);
    }
    return hash;
  });
  py::class_<at::hpu::HPUGraph>(m, "HPUGraph").def(pybind11::init());
  m.def(
      "capture_begin", [](at::hpu::HPUGraph& graph) { graph.capture_begin(); });
  m.def("capture_begin", [](at::hpu::HPUGraph& graph, bool dry_run) {
    graph.capture_begin(dry_run);
  });
  m.def("get_user_input_match_indices", [](at::hpu::HPUGraph& graph) {
    return graph.get_user_input_match_indices();
  });
  m.def("capture_end", [](at::hpu::HPUGraph& graph) { graph.capture_end(); });
  m.def("replay", [](at::hpu::HPUGraph& graph, bool async = false) {
    graph.replay(async);
  });
  m.def(
      "replayV2",
      [](at::hpu::HPUGraph& graph,
         std::vector<at::Tensor>& static_inputs,
         std::vector<at::Tensor>& inputs,
         bool async = false) { graph.replayV2(static_inputs, inputs, async); });
  m.def(
      "replayV3",
      [](at::hpu::HPUGraph& graph,
         std::vector<at::Tensor>& inputs,
         bool async = false) { graph.replayV3(inputs, async); });
  m.def("clear_inputs", [](at::hpu::HPUGraph& graph) { graph.clear_inputs(); });
  m.def(
      "mark_user_outputs",
      [](at::hpu::HPUGraph& graph, std::vector<at::Tensor>& outputs) {
        graph.mark_user_outputs(outputs);
      });
  m.def(
      "mark_user_inputs",
      [](at::hpu::HPUGraph& graph, std::vector<at::Tensor>& static_inputs) {
        graph.mark_user_inputs(static_inputs);
      });
  m.def("destroy", [](at::hpu::HPUGraph& graph) { graph.destroy(); });
  m.def("enable_dynamic_shape", []() {
    habana_helpers::EnableRefineDynamicShape();
  });
  m.def("disable_dynamic_shape", []() {
    habana_helpers::DisableRefineDynamicShape();
  });
  m.def("get_dynamic_shape_status", []() {
    return habana_helpers::GetRefineDynamicShapeStatus();
  });
  m.def("enable_optim_output_sif", []() {
    habana_helpers::EnableOpimDynamicOutputSIF();
  });
  m.def("disable_optim_output_sif", []() {
    habana_helpers::DisableOpimDynamicOutputSIF();
  });
  m.def(
      "enable_inference_mode", []() { habana_helpers::EnableInferenceMode(); });
  m.def("disable_inference_mode", []() {
    habana_helpers::DisableInferenceMode();
  });
  m.def("enable_quantization", []() { habana_helpers::EnableQuantization(); });
  m.def(
      "disable_quantization", []() { habana_helpers::DisableQuantization(); });
  m.def(
      "record_stream",
      [](at::Tensor tensor,
         int64_t stream_id,
         int64_t device_index,
         int64_t device_type) {
        auto stream = c10::hpu::HPUStream::unpack3(
            stream_id,
            static_cast<c10::DeviceIndex>(device_index),
            static_cast<c10::DeviceType>(device_type));

        habana::HPUDeviceAllocator::recordStream(
            tensor.storage().data_ptr(), stream);
      });
  m.def(
      "enable_const_section_serialization",
      [](const char* path, bool clear_path, bool use_compression) {
        habana_helpers::EnableConstSectionSerialization(
            path, clear_path, use_compression);
      });
  m.def("enable_matmul3d_2d_reshape", []() {
    habana_helpers::EnableMatmul3d2dReshape();
  });
  m.def("disable_matmul3d_2d_reshape", []() {
    habana_helpers::DisableMatmul3d2dReshape();
  });

  m.def("enable_recompute_FSDPA", [](bool recompute) {
    habana_helpers::enableRecomputeFSDPA(recompute);
  });
  m.def("is_recompute_FSDPA_enabled", []() {
    return habana_helpers::isRecomputeFSDPAEnabled();
  });

  m.def(
      "custom_op_calc_out_shape_no_params",
      [](const char* opname, const std::vector<at::Tensor>& inputs) {
        return habana::CustomOpOutShapeFunRegistrar::GetInstance().CalcOutShape(
            opname, inputs);
      });
  m.def(
      "custom_op_calc_out_shape_params_int",
      [](const char* opname,
         const std::vector<at::Tensor>& inputs,
         const std::vector<int64_t>& params) {
        return habana::CustomOpOutShapeFunRegistrar::GetInstance().CalcOutShape(
            opname, inputs, params);
      });
  m.def(
      "custom_op_calc_out_shape_params_float",
      [](const char* opname,
         const std::vector<at::Tensor>& inputs,
         const std::vector<float>& params) {
        return habana::CustomOpOutShapeFunRegistrar::GetInstance().CalcOutShape(
            opname, inputs, params);
      });

  pybind11::cpp_function eager_cleanup = []() {
    PT_EAGER_DEBUG("Eager cleanup.");
    habana::HPURegistrar::synchronize_host_multistage_pipeline();
  };

  pybind11::module::import("atexit").attr("register")(eager_cleanup);

  m.doc() = "This module registers hpu backend.";
}
