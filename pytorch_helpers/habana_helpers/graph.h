/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <synapse_helpers/device.h>
#include <synapse_helpers/graph.h>
#include <habana_device/hpu_cached_devices.h>


namespace habana_helpers {
static synapse_helpers::graph create_graph(int device_id, std::string name) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto graph_or_error = synapse_helpers::graph::create(device, name);

  if (absl::holds_alternative<synapse_helpers::synapse_error>(graph_or_error)) {
    auto error = absl::get<synapse_helpers::synapse_error>(graph_or_error);
    TORCH_CHECK(error.status, error.error);
  }
  return absl::get<synapse_helpers::graph>(std::move(graph_or_error));
}
} // namespace habana_helpers
