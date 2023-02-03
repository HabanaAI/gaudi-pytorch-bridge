#include <torch/extension.h>

#include "pytorch_helpers/habana_helpers/profiling/activity_profiler.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_start_activity_profiler", []() { habana::start_profiler_session(); });
  m.def("_stop_activity_profiler", []() { habana::stop_profiler_session(); });
  m.def(
      "_export_logs",
      [](const std::string& path) { habana::export_profiler_logs(path); },
      py::arg("path") = "");
  m.def(
      "_add_custom_tag_begin",
      [](const std::string& tag) { return habana::add_custom_tag_begin(tag); },
      py::arg("tag") = "");
  m.def("_add_custom_tag_end", [](uint64_t id) {
    habana::add_custom_tag_end(id);
  });
  m.doc() = "This module registers hpu hardware profiler API";
}