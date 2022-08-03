/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
// clang-format off
#include <pybind11/chrono.h>
#include <synapse_common_types.h>
#include <torch/extension.h>
#include <ATen/autocast_mode.h>
// clang-format on
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_device/HPUGraph.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/synapse_helpers/stream.h"

void hpu_init() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  device.get_count_by_current_type();
  // later will add device properties here.
}

const std::string get_device_name(int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return device.name();
}

const synapse_helpers::MemoryStats get_mem_stat(int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synapse_helpers::MemoryStats stats;
  device.get_device_memory().get_memory_stats(&stats);
  return stats;
}

void reset_peak_memory_stats(int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = synapse_helpers::HPURegistrar::get_device();
  device.get_device_memory().reset_peak_memory_stats();
}

void clear_memory_stats(int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = synapse_helpers::HPURegistrar::get_device();
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", []() { hpu_init(); });
  m.def("current_device", []() {
    auto& d = synapse_helpers::HPURegistrar::get_device();
    return d.id();
  });
  m.def("synchronize_device", []() {
    synapse_helpers::HPURegistrar::synchronize_device();
  });
  m.def("device_count", []() {
    return synapse_helpers::HPURegistrar::get_total_device_count();
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
  m.def("get_device_name", [](int id) { return get_device_name(id); });
  py::class_<HPUStream>(m, "HPUStream");
  m.def("get_stream", [](bool isHighPriorityStream, int device) {
    HPUStream stream = getStreamFromPool(isHighPriorityStream, device);
    return stream;
  });
  m.def("query", [](HPUStream stream) {
    bool finished = stream.query();
    return finished;
  });
  m.def("synchronize", [](HPUStream stream) {
    stream.synchronize(); // TBD: release GIL  ?
  });
  m.def("get_current_stream", []() {
    HPUStream stream = getCurrentHPUStream();
    return stream;
  });
  m.def("set_current_stream", [](HPUStream stream) {
    setCurrentHPUStream(stream);
  });
  m.def("get_default_stream", []() {
    HPUStream stream = getDefaultHPUStream();
    return stream;
  });
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

  py::class_<at::hpu::HPUGraph>(m, "HPUGraph").def(pybind11::init());
  m.def(
      "capture_begin", [](at::hpu::HPUGraph& graph) { graph.capture_begin(); });
  m.def("capture_end", [](at::hpu::HPUGraph& graph) { graph.capture_end(); });
  m.def("replay", [](at::hpu::HPUGraph& graph) { graph.replay(); });
  m.doc() = "This module registers hpu backend.";
}
