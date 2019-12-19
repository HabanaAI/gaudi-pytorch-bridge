#pragma once

#include <c10/util/ArrayRef.h>
#include <synapse/include/synapse_api.h>
#include <string>
#include <unordered_map>

#include "synapse/include/synapse_api_types.h"

namespace habana_helpers {
std::string unique_recipe_name_generator(std::string recipe_name);

synTensorDescriptorTr synapse_tensor_descriptor_builder(
    const c10::IntArrayRef& shape,
    const synDataType dtype,
    const std::string& name,
    const bool persistent);

void compile_and_run(
    const std::string& recipe_prefix,
    const synGraphHandle graph_handle,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    const uint32_t device_id);
} // namespace habana_helpers