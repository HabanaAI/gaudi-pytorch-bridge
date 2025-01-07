/**
* Copyright (c) 2021-2024 Intel Corporation
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
#pragma once

#include <functional>
#include <string>
#include <vector>
#include "backend/synapse_helpers/graph.h"

namespace synapse_helpers {
class device;

class recipe {
 public:
  explicit recipe(device& device) : device_{device} {}
  recipe(const recipe&) = delete;
  recipe(recipe&&) = delete;
  recipe& operator=(const recipe&) = delete;
  recipe& operator=(recipe&&) = delete;
  bool create(synapse_helpers::graph& graph);
  void set_inputs_outputs_names(
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names);
  void populate_syn_tensor_ids();
  void launch(
      const std::vector<void*>& in_buffers,
      const std::vector<void*>& out_buffers,
      std::unique_ptr<device_ptr_lock>& addr_locked,
      stream& compute_stream);
  std::shared_ptr<synapse_helpers::graph::recipe_handle> getRecipeHandle() {
    return recipe_handle_;
  }

 private:
  std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_handle_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  uint64_t workspace_size_{0};
  device& device_;

  std::unique_ptr<uint64_t[]> tensor_ids;
  std::vector<const char*> tensor_names;
};
} // namespace synapse_helpers
