/**
 * Copyright (c) 2021-2025 Intel Corporation
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
  auto& synapse_device = habana::HPUDeviceContext::get_device();
  if (synapse_device.type() == synDeviceGaudi2) {
    habana_helpers::EnableInferenceMode();
    habana_helpers::EnableQuantization();
    synapse_device.set_is_dynamic_quantization(true);
    synapse_helpers::graph graph = habana_helpers::create_graph(
        synapse_device.id(), "attributesTestGraph");
    auto handle = graph.get_graph_handle();
    ASSERT_NE(handle, nullptr);
    std::vector<synGraphAttributeVal> getValues(3);
    synGraphAttribute att[] = {
        GRAPH_ATTRIBUTE_INFERENCE,
        GRAPH_ATTRIBUTE_QUANTIZATION,
        GRAPH_ATTRIBUTE_USE_DYNAMIC_QUANTIZATION};
    ASSERT_EQ(
        synSuccess,
        synGraphGetAttributes(handle, att, getValues.data(), getValues.size()));
    ASSERT_EQ(getValues[0].iAttrVal, 1);
    ASSERT_EQ(getValues[1].iAttrVal, 1);
    ASSERT_EQ(getValues[2].iAttrVal, 1);
    habana_helpers::DisableInferenceMode();
    habana_helpers::DisableQuantization();
    synapse_device.set_is_dynamic_quantization(false);
  }
}
