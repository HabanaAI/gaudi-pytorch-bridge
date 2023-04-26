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
#include <synapse_api.h>
#include <synapse_common_types.h>

#include <absl/types/variant.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/event.h"
#include "backend/synapse_helpers/recipe.h"
#include "backend/synapse_helpers/synapse_error.h"
#include "habana_helpers/logging.h"

namespace synapse_helpers {
recipe::recipe(device& device) : device_{device} {}

bool recipe::create(synapse_helpers::graph& graph) {
  auto&& compile_result{graph.compile()};
  if (ABSL_PREDICT_FALSE(
          absl::holds_alternative<synapse_helpers::synapse_error>(
              compile_result))) {
    auto& error = absl::get<synapse_helpers::synapse_error>(compile_result);
    PT_SYNHELPER_FATAL(
        "syn compile encountered : ",
        error.error,
        " ",
        Logger::formatStatusMsg(error.status));
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
            "syn query workspace failed: ",
            error.error,
            " ",
            Logger::formatStatusMsg(error.status));
      }
      workspace_size_ = get_value(ws_size_result);
    }
  }
  return (recipe_handle != nullptr);
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
  populate_syn_tensor_ids();
}

void recipe::populate_syn_tensor_ids() {
  if (nullptr == tensor_ids) {
    auto num_tensors = input_names_.size() + output_names_.size();
    tensor_ids = new uint64_t[num_tensors];
    tensor_names = new const char*[num_tensors];

    size_t tensor_idx{0};
    for (const auto& n : input_names_) {
      tensor_names[tensor_idx++] = n.c_str();
    }
    for (const auto& n : output_names_) {
      tensor_names[tensor_idx++] = n.c_str();
    }

    synStatus status = synTensorRetrieveIds(
        recipe_handle_->syn_recipe_handle_,
        tensor_names,
        tensor_ids,
        num_tensors);

    if (ABSL_PREDICT_FALSE(status != synStatus::synSuccess)) {
      PT_SYNHELPER_FATAL(
          Logger::formatStatusMsg(status),
          "synTensorRetrieveIds launch failed");
    }
  }
}

bool recipe::launch(
    const std::vector<void*>& in_buffers,
    const std::vector<void*>& out_buffers,
    std::unique_ptr<device_ptr_lock>& addr_locked,
    stream& compute_stream) {
  std::vector<synLaunchTensorInfoExt> syn_info;
  syn_info.reserve(input_names_.size() + output_names_.size());

  size_t tensor_idx{0};
  for (size_t i = 0; i < input_names_.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfoExt{
        input_names_[i].c_str(),
        reinterpret_cast<uint64_t>(in_buffers[i]),
        DATA_TENSOR,
        {0},
        tensor_ids[tensor_idx++]});
  for (size_t i = 0; i < output_names_.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfoExt{
        output_names_[i].c_str(),
        reinterpret_cast<uint64_t>(out_buffers[i]),
        DATA_TENSOR,
        {0},
        tensor_ids[tensor_idx++]});

  std::vector<shared_event> ext_events;
  auto&& error_optional{synapse_helpers::graph::launch(
      device_,
      *recipe_handle_,
      workspace_size_,
      syn_info,
      addr_locked,
      ext_events,
      compute_stream)};
  if (ABSL_PREDICT_FALSE(error_optional.has_value())) {
    auto& error = error_optional.value();
    PT_SYNHELPER_FATAL(
        "syn launch encountered : ",
        error.error,
        " ",
        Logger::formatStatusMsg(error.status));
    return false;
  }
  return true;
}

std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe::
    getRecipeHandle() {
  return recipe_handle_;
}

recipe::~recipe() {
  if (nullptr != tensor_names) {
    delete[] tensor_ids;
    delete[] tensor_names;
  }
}
} // namespace synapse_helpers
