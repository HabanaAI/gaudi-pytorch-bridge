/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#pragma once

#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/graph.h"
#include "habana_helpers/logging.h"

namespace habana_helpers {
static inline synapse_helpers::graph create_graph(
    int device_id,
    std::string name,
    bool dry_run = false,
    bool eager_mode = false) {
  auto& device = habana::HPURegistrar::get_device(device_id);
  return synapse_helpers::graph::create(
      device.syn_device(), name, dry_run, eager_mode);
}
} // namespace habana_helpers
