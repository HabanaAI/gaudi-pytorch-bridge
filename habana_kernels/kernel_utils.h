#pragma once

#include <c10/util/ArrayRef.h>
#include <synapse/include/synapse_api.h>
#include <synapse/include/synapse_api_types.h>
#include <synapse_helpers/habana_tensor.h>
#include <string>
#include <unordered_map>

namespace habana_helpers {
std::string unique_recipe_name_generator(std::string recipe_name);

void compile_and_run(
    const std::string& recipe_prefix,
    const synGraphHandle graph_handle,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    const uint32_t device_id);
} // namespace habana_helpers