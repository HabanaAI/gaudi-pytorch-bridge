/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include <gtest/gtest.h>
#include <synapse_api_types.h>
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/graph.h"

TEST(SynapseHelpersGraphTest, graphAttributes) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& synapse_device = habana::HPURegistrar::get_device().syn_device();
  if (synapse_device.type() == synDeviceGaudi2) {
    habana_helpers::EnableInferenceMode();
    habana_helpers::EnableQuantization();
    synapse_helpers::graph graph = habana_helpers::create_graph(
        synapse_device.id(), "attributesTestGraph");
    auto handle = graph.get_graph_handle();
    ASSERT_NE(handle, nullptr);
    std::vector<uint64_t> getValues = {0, 0};
    synGraphAttribute att[] = {
        GRAPH_ATTRIBUTE_INFERENCE, GRAPH_ATTRIBUTE_QUANTIZATION};
    ASSERT_EQ(
        synSuccess,
        synGraphGetAttribute(handle, att, getValues.data(), getValues.size()));
    ASSERT_EQ(getValues[0], 1);
    ASSERT_EQ(getValues[1], 1);
    habana_helpers::DisableInferenceMode();
    habana_helpers::DisableQuantization();
  }
}
