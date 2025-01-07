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

#include "pytorch_helpers/synapse_logger/synapse_logger.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ///////////////// Host Profiler API //////////////////
  m.def("setup_profiler", []() {
    std::string cmd = "stop_data_capture";
    synapse_logger::command(cmd);
    cmd = "no_eager_flush";
    synapse_logger::command(cmd);
    cmd = "use_pid_suffix";
    synapse_logger::command(cmd);
    cmd = "optimize_trace";
    synapse_logger::command(cmd);
  });

  m.def("start_profiler", []() {
    std::string cmd = "restart";
    synapse_logger::command(cmd);
  });

  m.def("stop_profiler", []() {
    std::string cmd = "disable_mask";
    synapse_logger::command(cmd);
  });

  m.doc() = "This module registers hpu host profiler API";
}
