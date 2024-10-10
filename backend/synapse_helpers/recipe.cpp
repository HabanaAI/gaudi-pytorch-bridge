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

bool recipe::create(synapse_helpers::graph& graph) {
  auto recipe_handle = graph.compile();
  if (recipe_handle != nullptr) {
    recipe_handle_ = recipe_handle;
    if (!graph.is_empty()) {
      // first time, we need to get workspace size of the recipe, that was
      // compiled
      workspace_size_ =
          synapse_helpers::graph::query_workspace_size(*recipe_handle_);
    }
  }
  return (recipe_handle != nullptr);
}

void recipe::set_inputs_outputs_names(
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {
  for (auto&& input : input_names) {
    input_names_.emplace_back(input);
  }
  for (auto&& output : output_names) {
    output_names_.emplace_back(output);
  }
  populate_syn_tensor_ids();
}

void recipe::populate_syn_tensor_ids() {
  if (!tensor_ids) {
    auto num_tensors = input_names_.size() + output_names_.size();
    tensor_ids = std::make_unique<uint64_t[]>(num_tensors);
    tensor_names.reserve(num_tensors);

    for (const auto& n : input_names_) {
      tensor_names.emplace_back(n.c_str());
    }
    for (const auto& n : output_names_) {
      tensor_names.emplace_back(n.c_str());
    }

    synStatus status = synTensorRetrieveIds(
        recipe_handle_->syn_recipe_handle_,
        tensor_names.data(),
        tensor_ids.get(),
        num_tensors);

    if (ABSL_PREDICT_FALSE(status != synStatus::synSuccess)) {
      PT_SYNHELPER_FATAL(
          Logger::formatStatusMsg(status),
          "synTensorRetrieveIds launch failed");
    }
  }
}

void recipe::launch(
    const std::vector<void*>& in_buffers,
    const std::vector<void*>& out_buffers,
    std::unique_ptr<device_ptr_lock>& addr_locked,
    stream& compute_stream) {
  std::vector<synLaunchTensorInfo> syn_info;
  syn_info.reserve(input_names_.size() + output_names_.size());

  size_t tensor_idx{0};
  for (size_t i = 0; i < input_names_.size(); ++i) {
    syn_info.emplace_back(synLaunchTensorInfo{
        input_names_[i].c_str(),
        reinterpret_cast<uint64_t>(in_buffers[i]),
        DATA_TENSOR,
        {0},
        tensor_ids.get()[tensor_idx++]});
  }
  for (size_t i = 0; i < output_names_.size(); ++i) {
    syn_info.emplace_back(synLaunchTensorInfo{
        output_names_[i].c_str(),
        reinterpret_cast<uint64_t>(out_buffers[i]),
        DATA_TENSOR,
        {0},
        tensor_ids.get()[tensor_idx++]});
  }

  std::vector<shared_event> ext_events;
  synapse_helpers::graph::launch(
      device_,
      *recipe_handle_,
      workspace_size_,
      syn_info,
      addr_locked,
      ext_events,
      compute_stream);
}

} // namespace synapse_helpers
