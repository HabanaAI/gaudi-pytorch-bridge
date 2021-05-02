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

#include <functional>
#include <string>
#include <vector>
#include "synapse_helpers/graph.h"

namespace synapse_helpers {
class device;

class recipe {
 public:
  explicit recipe(device& device);
  recipe(const recipe&) = delete;
  recipe(recipe&&) = delete;
  recipe& operator=(const recipe&) = delete;
  recipe& operator=(recipe&&) = delete;
  bool create(synapse_helpers::graph& graph);
  void set_inputs_outputs_names(
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names);
  bool launch(
      const std::vector<void*>& in_buffers,
      const std::vector<void*>& out_buffers);
  std::shared_ptr<synapse_helpers::graph::recipe_handle> getRecipeHandle();
  ~recipe();

 private:
  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_handle_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  uint64_t workspace_size_{0};
  device& device_;
};
} // namespace synapse_helpers
