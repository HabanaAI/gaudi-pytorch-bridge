#pragma once

#include <c10/util/ArrayRef.h>
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
} // namespace habana_helpers