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