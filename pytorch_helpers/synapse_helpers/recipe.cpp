/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <synapse_api.h>
#include <synapse_common_types.h>

#include <absl/types/variant.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include "habana_helpers/logging.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/event.h"
#include "synapse_helpers/recipe.h"
#include "synapse_helpers/synapse_error.h"

namespace synapse_helpers {
recipe::recipe(device& device) : device_{device} {}

bool recipe::create(synapse_helpers::graph& graph) {
  auto&& compile_result{graph.compile()};
  if (ABSL_PREDICT_FALSE(
          absl::holds_alternative<synapse_helpers::synapse_error>(
              compile_result))) {
    auto& error = absl::get<synapse_helpers::synapse_error>(compile_result);
    PT_SYNHELPER_FATAL(
        "syn compile encountered : ", error.error, " ", error.status);
  }
  auto recipe_handle = get_value(std::move(compile_result));
  if (recipe_handle != nullptr) {
    recipe_handle_ = recipe_handle;
    if (!graph.is_empty()) {
      // first time, we need to get workspace size of the recipe, that was
      // compiled
      auto&& ws_size_result{
          synapse_helpers::graph::query_workspace_size(*recipe_handle_)};
      if (ABSL_PREDICT_FALSE(
              absl::holds_alternative<synapse_helpers::synapse_error>(
                  ws_size_result))) {
        auto& error = absl::get<synapse_helpers::synapse_error>(ws_size_result);
        PT_SYNHELPER_FATAL(
            "syn query workspace failed: ", error.error, " ", error.status);
      }
      workspace_size_ = get_value(ws_size_result);
    }
  }
  return (recipe_handle != nullptr);
}

std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe::
    getRecipeHandle() {
  return recipe_handle_;
}

void recipe::set_inputs_outputs_names(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {
  for (auto input : input_names) {
    input_names_.emplace_back(std::move(input));
  }
  for (auto output : output_names) {
    output_names_.emplace_back(std::move(output));
  }
}

bool recipe::launch(
    const std::vector<void*>& in_buffers,
    const std::vector<void*>& out_buffers) {
  std::vector<synLaunchTensorInfo> syn_info;
  syn_info.reserve(input_names_.size() + output_names_.size());
  for (size_t i = 0; i < input_names_.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        input_names_[i].c_str(),
        reinterpret_cast<uint64_t>(in_buffers[i]),
        DATA_TENSOR,
        {0}});
  for (size_t i = 0; i < output_names_.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        output_names_[i].c_str(),
        reinterpret_cast<uint64_t>(out_buffers[i]),
        DATA_TENSOR,
        {0}});

  auto&& error_optional{synapse_helpers::graph::launch(
      device_, *recipe_handle_, workspace_size_, syn_info)};
  if (ABSL_PREDICT_FALSE(error_optional.has_value())) {
    auto& error = error_optional.value();
    PT_SYNHELPER_FATAL(
        "syn launch encountered : ", error.error, " ", error.status);
    return false;
  }
  return true;
}

recipe::~recipe() = default;
} // namespace synapse_helpers
