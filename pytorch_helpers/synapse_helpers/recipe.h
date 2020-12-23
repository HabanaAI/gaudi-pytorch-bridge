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

class recipe {
 public:
  explicit recipe();
  recipe(const recipe&) = delete;
  recipe(recipe&&) = delete;
  recipe& operator=(const recipe&) = delete;
  recipe& operator=(recipe&&) = delete;
  bool create(synapse_helpers::graph& graph);
  void create_launch_info();
  void set_inputs_outputs_names(
      std::vector<std::string> input_names,
      std::vector<std::string> output_names);
  bool launch(
      const std::vector<void*>& in_buffers,
      const std::vector<void*>& out_buffers);
  std::shared_ptr<synapse_helpers::graph::recipe_handle> getRecipeHandle();
  ~recipe();

 private:
  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_handle_;
  absl::optional<synapse_helpers::graph::launch_info> launch_info_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};
} // namespace synapse_helpers
