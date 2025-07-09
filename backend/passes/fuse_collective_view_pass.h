/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/passes/hpu_habana_pass_interface.h"

#include "habana_helpers/logging.h"

namespace habana {
class HabanaLaunchOpPT;
class FuseCollectiveViewPassData;

struct ExternalParams {
  uint64_t offset;
  uint64_t numel;
};

// Derived class
class FuseCollectiveViewPass : public JITGraphPass<FuseCollectiveViewPassData> {
 public:
  FuseCollectiveViewPass(HabanaLaunchOpPT* habana_launch_op_ptr)
      : habana_launch_op_ptr_(habana_launch_op_ptr) {}

  virtual ~FuseCollectiveViewPass() = default;

  std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>&
  getInputValPtrToParamsMap() {
    return input_valptr_to_params_map_;
  }

  std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>&
  getOutputValPtrToParamsMap() {
    return output_valptr_to_params_map_;
  }

  std::shared_ptr<torch::jit::Graph>& getOriginalGraph();

  std::shared_ptr<torch::jit::Graph>& getClonedGraph() {
    return cloned_graph_;
  }

  std::unique_ptr<FuseCollectiveViewPassData> VisitGraph(
      const std::shared_ptr<torch::jit::Graph> graph);

  void PostRunFuseOpsPasses(
      torch::jit::Node* node,
      habana_helpers::CollectiveKernelInfos::Info& kernel_info);

 private:
  std::string pass_name_ = "fuse_collective_slice_view_pass";

  HabanaLaunchOpPT* habana_launch_op_ptr_ = nullptr;
  std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>
      input_valptr_to_params_map_;
  std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>
      output_valptr_to_params_map_;
  std::shared_ptr<torch::jit::Graph> original_graph_;
  std::shared_ptr<torch::jit::Graph> cloned_graph_;

  void RunFuseOpsPasses(const std::shared_ptr<torch::jit::Graph> graph);
  void RunFuseOps(torch::jit::Node* collective_node, int index);
  bool RunFuseOps(
      torch::jit::graph_node_list graph_nodes,
      bool is_check_mode = false);
  void FuseSliceInsertOps(
      torch::jit::Node* collective_node,
      torch::jit::Value* output,
      torch::jit::Node* slice_insert_node,
      std::vector<torch::jit::Node*>& const_node_vec);
  void FuseSliceOps(torch::jit::Node* slice_node);
  void FuseSqueezeViewOps(torch::jit::Node* node);

  torch::jit::Value* GetInputValue(
      torch::jit::Node* node,
      bool is_node_output = true);
  void RelocateJITStack(
      CValuePtrToIValuePtrMap& value_to_ivalue,
      std::shared_ptr<torch::jit::Graph>& graph);
  void PrepareJITStack(CValuePtrToIValuePtrMap& value_to_ivalue);
  void RestoreJITStack(CValuePtrToIValuePtrMap& value_to_ivalue);
  bool CanFuse(CValPtr value, int64_t dim = 0, int64_t step = 1);
  bool IsGraphInputOutput(
      torch::jit::Value* value,
      bool is_graph_output = false);
  bool CanFuse(torch::jit::Node* node, bool is_node_output = false);
  bool NeedCheck(std::shared_ptr<torch::jit::Graph> graph);
  void GetExternalParams(
      CValPtr value,
      int64_t dim,
      int64_t start,
      int64_t end,
      ExternalParams& params);
  std::shared_ptr<torch::jit::Graph> CreateClonedGraph(
      std::shared_ptr<torch::jit::Graph> graph);
  void PatchPTTensorInfo(
      CValPtr value,
      size_t item_size,
      PtTensorInfoShared& ti,
      std::shared_ptr<ExternalParams> params_ptr,
      bool is_input = true);
  void ProcessInputPTTensorInfo(
      std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>&
          valptr_to_params_map,
      torch::jit::Node* node,
      habana_helpers::CollectiveKernelInfos::Info& kernel_info);
  void ProcessOutputPTTensorInfo(
      std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>&
          valptr_to_params_map,
      torch::jit::Node* node,
      habana_helpers::CollectiveKernelInfos::Info& kernel_info);
};

// Pass data
class FuseCollectiveViewPassData {
 public:
  FuseCollectiveViewPassData(std::shared_ptr<FuseCollectiveViewPass> pass)
      : pass_(pass) {}

  std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>&
  getInputValPtrToParamsMap() {
    return pass_->getInputValPtrToParamsMap();
  }

  std::unordered_map<CValPtr, std::shared_ptr<ExternalParams>>&
  getOutputValPtrToParamsMap() {
    return pass_->getOutputValPtrToParamsMap();
  }

  std::shared_ptr<torch::jit::Graph>& getOriginalGraph() {
    return pass_->getOriginalGraph();
  }

  std::shared_ptr<torch::jit::Graph>& getClonedGraph() {
    return pass_->getClonedGraph();
  }

  void PostRunFuseOpsPasses(
      torch::jit::Node* node,
      habana_helpers::CollectiveKernelInfos::Info& kernel_info) {
    pass_->PostRunFuseOpsPasses(node, kernel_info);
  }

 private:
  std::shared_ptr<FuseCollectiveViewPass> pass_;
};
} // namespace habana
