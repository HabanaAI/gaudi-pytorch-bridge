#include <torch/extension.h>

#include "pytorch_helpers/habana_helpers/profiling/activity_profiler.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_start_activity_profiler", []() { habana::start_profiler_session(); });
  m.def("_stop_activity_profiler", []() { habana::stop_profiler_session(); });
  m.def(
      "_export_logs",
      [](const std::string& path) { habana::export_profiler_logs(path); },
      py::arg("path") = "");
  m.doc() = "This module registers hpu hardware profiler API";
}