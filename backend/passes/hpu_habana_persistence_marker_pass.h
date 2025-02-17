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

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_set>

#include <ATen/Tensor.h>
#include <absl/hash/hash.h>
#include <absl/types/variant.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/passes/hpu_habana_pass_interface.h"

#include "habana_helpers/logging.h"

namespace habana {
// Pass data
class PersistenceMarkerPassData {
 public:
  PersistenceMarkerPassData(
      std::unordered_map<CValPtr, bool> valptr_to_persistent_map,
      std::unordered_map<CValPtr, bool> valptr_to_external_map)
      : valptr_to_persistent_map_(valptr_to_persistent_map),
        valptr_to_external_map_(valptr_to_external_map) {}

  bool IsPersistentNode(CValPtr value_ptr) {
    if (valptr_to_persistent_map_.find(value_ptr) !=
        valptr_to_persistent_map_.end()) {
      return valptr_to_persistent_map_[value_ptr];
    } else {
      return false;
    }
  }

  bool IsExternalNode(CValPtr value_ptr) {
    if (valptr_to_external_map_.find(value_ptr) !=
        valptr_to_external_map_.end()) {
      return valptr_to_external_map_[value_ptr];
    } else {
      return false;
    }
  }

 private:
  std::unordered_map<CValPtr, bool> valptr_to_persistent_map_;
  std::unordered_map<CValPtr, bool> valptr_to_external_map_;
};

// Derived class
class PersistenceMarkerPass : public JITGraphPass<PersistenceMarkerPassData> {
 public:
  PersistenceMarkerPass(HabanaLaunchOpPT* habana_launch_op_ptr)
      : habana_launch_op_ptr_(habana_launch_op_ptr) {}
  std::unique_ptr<PersistenceMarkerPassData> VisitGraph(
      const std::shared_ptr<torch::jit::Graph> graph);

 private:
  std::string pass_name_ = "persistence_marker_pass";

  std::unordered_map<CValPtr, bool> valptr_to_persistent_map_;
  std::unordered_map<CValPtr, bool> valptr_to_external_map_;

  HabanaLaunchOpPT* habana_launch_op_ptr_ = NULL;

  /* Guideline: Accessors and mutators (get and set functions) may be named like
   * variables. */
  void set_persistence_input(torch::jit::Node*, int inputId);
  void set_persistence_output(torch::jit::Node*, int outputId);
  void set_external_input(torch::jit::Node*);

  void MarkPersistenceNodes(torch::jit::graph_node_list graph_nodes);
  void MarkProducerExternal(torch::jit::Value* val);
  void ExternalMarkingPass(torch::jit::graph_node_list graph_nodes);
  void RunMetaDataAdjustmentPasses(torch::jit::graph_node_list graph_nodes);
  void HandleSpecialOps(
      torch::jit::Node*,
      const std::vector<std::string>& ignoreOpsList,
      int inputId);
};

} // namespace habana
