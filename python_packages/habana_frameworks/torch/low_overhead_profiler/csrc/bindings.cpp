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
#include "backend/habana_device/HPUDevice.h"
#include "backend/synapse_helpers/env_flags.h"
#include "pybind11/stl.h"
#include "pytorch_helpers/low_overhead_profiler/profiler.h"
#include "synapse_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "This module registers low overhead host profiler API";

  m.def("_start_lo_host_profiler", []() {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_METRICS_COLLECTION, true, 1);
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_TRACES_COLLECTION, true, 1);
  });
  m.def("_stop_lo_host_profiler", []() {
    habana::HPUDeviceContext::synchronize_host_multistage_pipeline();
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_METRICS_COLLECTION, false, 1);
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_TRACES_COLLECTION, false, 1);
  });
  m.def("_flush_lo_host_profiler", []() {
    LOP::ProfilerEngine::get_inst().flush();
  });
}