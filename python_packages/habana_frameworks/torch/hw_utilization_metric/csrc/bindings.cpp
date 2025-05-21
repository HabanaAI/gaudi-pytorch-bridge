/**
 * Copyright (c) 2025 Intel Corporation
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

#include "backend/synapse_helpers/utilization_metrics.h"

#include <torch/extension.h>
#include "pybind11/stl.h"

/**
 * UtilizationMetrics provides a comprehensive API for monitoring hardware
 * utilization.
 *
 * This module integrates two sources of metrics:
 *   - Device-level metrics (via HL-SMI) indicating the average HPU usage.
 *   - Event synchronization metrics indicating the balance between active
 * processing and idle time.
 *
 * The exposed functions allow you to:
 *   - Set the HL-SMI library path.
 *   - Start tracking hardware utilization metrics.
 *   - Stop tracking the metrics.
 *   - Reset the counters.
 *   - Retrieve the current metrics as a Python dictionary.
 */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "_start_hw_utilization_metrics",
      []() { synapse_helpers::UtilizationMetrics::getInstance().start(); },
      "Starts tracking hardware utilization metrics, including both HL-SMI and event synchronization metrics.");

  m.def(
      "_resume_hw_utilization_metrics",
      []() { synapse_helpers::UtilizationMetrics::getInstance().resume(); },
      "Resumes tracking hardware utilization metrics, including both HL-SMI and event synchronization metrics.");

  m.def(
      "_stop_hw_utilization_metrics",
      []() { synapse_helpers::UtilizationMetrics::getInstance().stop(); },
      "Stops tracking hardware utilization metrics.");

  m.def(
      "_reset_hw_utilization_metrics",
      []() { synapse_helpers::UtilizationMetrics::getInstance().reset(); },
      "Resets the hardware utilization counters.");

  m.def(
      "_get_hw_utilization_metrics",
      []() {
        auto counters =
            synapse_helpers::UtilizationMetrics::getInstance().getUtilization();
        py::dict result;
        result["hl_smi"] = counters.first;
        result["esync"] = counters.second;
        return result;
      },
      "Retrieves the current hardware utilization metrics as a dictionary with keys:\n"
      "  - 'hl_smi': Average HPU usage percentage.\n"
      "  - 'esync': Event synchronization ratio.");
}
