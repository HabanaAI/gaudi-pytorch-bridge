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
#pragma once

#include "backend/habana_device/HPUDevice.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/graph.h"
#include "habana_helpers/logging.h"

namespace habana_helpers {
static inline synapse_helpers::graph create_graph(
    int device_id,
    std::string name,
    bool dry_run = false,
    bool eager_mode = false) {
  auto& device = habana::HPUDeviceContext::get_device(device_id);
  return synapse_helpers::graph::create(device, name, dry_run, eager_mode);
}
} // namespace habana_helpers
