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

namespace habana_helpers {
static synapse_helpers::graph create_graph(
    synapse_helpers::device& device,
    std::string name) {
  auto graph_or_error = synapse_helpers::graph::create(device, name);

  if (absl::holds_alternative<synapse_helpers::synapse_error>(graph_or_error)) {
    auto error = absl::get<synapse_helpers::synapse_error>(graph_or_error);
    TORCH_HABANA_CHECK(error.status, error.error);
  } else {
    return absl::get<synapse_helpers::graph>(std::move(graph_or_error));
  }
}
} // namespace habana_helpers