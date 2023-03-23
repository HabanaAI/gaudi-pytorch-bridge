#include <torch/extension.h>
#include "pybind11/stl.h"

#include "backend/profiling/activity_profiler.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_start_activity_profiler", []() {
    habana::profile::start_profiler_session();
  });
  m.def("_stop_activity_profiler", []() {
    habana::profile::stop_profiler_session();
  });
  m.def(
      "_setup_activity_profiler_sources",
      [](bool synapse_logger,
         bool bridge,
         bool memory,
         std::vector<std::string> mandatory_events) {
        habana::profile::setup_profiler_sources(
            synapse_logger, bridge, memory, mandatory_events);
      },
      py::arg("synapse_logger") = "",
      py::arg("bridge") = "",
      py::arg("memory") = "",
      py::arg("mandatory_events") = "");
  m.def(
      "_export_logs",
      [](const std::string& path) {
        habana::profile::export_profiler_logs(path);
      },
      py::arg("path") = "");
  m.doc() = "This module registers hpu hardware profiler API";
}