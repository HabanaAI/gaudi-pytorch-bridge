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

#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/event.h"

namespace synapse_helpers {

class graph {
 public:
  graph() = delete;
  ~graph();
  graph(const graph&) = delete;
  graph& operator=(const graph&) = delete;
  graph(graph&&) noexcept;
  graph& operator=(graph&&) = delete;

  static graph create(
      device& device,
      std::string name,
      bool dry_run = false,
      bool eager_mode = false);

  static graph create_for_refinement(device& device, std::string name);

  static std::tuple<
      graph,
      std::vector<synTensorHandleMap>,
      std::vector<synNodeHandleMap>>
  duplicate(graph& other);

  bool inferShapes();

  static void getTensorGeometry(
      synTensor tensor_handle,
      std::vector<int64_t>& shape);

  static void setTensorGeometry(
      synTensor tensor_handle,
      std::vector<int64_t> shape);

  static void setTensorPermutation(
      synTensor tensor_handle,
      std::vector<uint8_t>& permute_or_empty);

  static void setTensorSectionOffset(synTensor tensor_handle, uint64_t offset);

  static void getNodeParams(
      const synGraphHandle graph_handle,
      const synNodeId node_id,
      void* node_params,
      unsigned* params_size);

  static void setNodeParams(
      const synGraphHandle graph_handle,
      const synNodeId node_id,
      const void* node_params,
      const unsigned params_size);

  void add_node(
      std::vector<synTensor>&& inputs,
      std::vector<synTensor>&& outputs,
      void* const params,
      const unsigned params_size,
      const std::string& node_type,
      synNodeId* ret_node_id,
      const char** input_layouts,
      const char** output_layouts,
      bool deterministic,
      const std::string& hints_str = "");

  template <typename ParamsT>
  void add_node(
      std::vector<synTensor>&& inputs,
      std::vector<synTensor>&& outputs,
      ParamsT* const params,
      const std::string& node_type,
      const char** input_layouts,
      const char** output_layouts,
      bool deterministic,
      const std::string& hints_str = "") {
    return add_node(
        std::move(inputs),
        std::move(outputs),
        params,
        sizeof(*params),
        node_type,
        nullptr,
        input_layouts,
        output_layouts,
        deterministic,
        hints_str);
  }

  synStatus set_synapse_control_edges_pt(
      std::vector<synNodeId>,
      std::vector<synNodeId>);

  bool is_empty() const {
    return graph_is_empty_;
  }

  void set_is_empty_value(bool flag) {
    graph_is_empty_ = flag;
  }

  struct recipe_handle {
    synRecipeHandle syn_recipe_handle_{nullptr};
    std::string recipe_name_{""};
    bool graph_is_empty_{false};
    bool in_execution_phase_{false};
    uint64_t get_recipe_host_mem_size();

    explicit recipe_handle(){};
    ~recipe_handle();

    recipe_handle& operator=(const recipe_handle&) = delete;
    recipe_handle(recipe_handle&&) = delete;
    recipe_handle& operator=(recipe_handle&&) = delete;

   private:
    uint64_t recipe_size_ = 0;
  };

  std::shared_ptr<recipe_handle> compile();

  struct OpNameContext {
    OpNameContext(graph& graph, const std::string& opName) : graph_(graph) {
      graph_.current_op_name_ = opName;
    }
    ~OpNameContext() {
      graph_.current_op_name_.reset();
    }
    graph& graph_;
  };

  friend OpNameContext;

  void add_control_edge(
      const std::string& src_node_name,
      const std::string& dst_node_name) {
    control_edges_container_[src_node_name].emplace(dst_node_name);
  };

  void add_data_edge(
      const std::string& src_node_name,
      const std::string& dst_node_name) {
    data_edges_container_[src_node_name].insert(dst_node_name);
  };

  static std::string_view name_suffix_from_type(
      synDataType type,
      bool use_int64 = false);

  static void query_recipe_tensor_info(
      const graph::recipe_handle& recipe_handle,
      std::vector<synRetrievedLaunchTensorInfo>& tensor_info_vec);

  static uint64_t query_workspace_size(
      const graph::recipe_handle& recipe_handle);

  static void launch(
      device& device,
      const graph::recipe_handle& recipe_handle,
      uint64_t workspace_size,
      std::vector<synLaunchTensorInfo>&& inputs_and_outputs_info,
      std::unique_ptr<device_ptr_lock>& address_lock,
      std::vector<shared_event>& ext_events,
      stream& compute_stream,
      size_t active_graph_key = 0);

  static void launch(
      device& device,
      const graph::recipe_handle& recipe_handle,
      uint64_t workspace_size,
      std::vector<synLaunchTensorInfo>& inputs_and_outputs_info,
      std::unique_ptr<device_ptr_lock>& address_lock,
      std::vector<shared_event>& ext_events,
      stream& compute_stream,
      size_t active_graph_key = 0);

  const std::string& name() const {
    return name_;
  }
  synGraphHandle get_graph_handle() const {
    return graph_handle_;
  }

  device& get_device() {
    return device_;
  }

  std::vector<synNodeId>& get_node_indices() {
    return op_to_node_container_pt_["jit_node"];
  }

  synNodeId get_node_index(size_t idx) {
    return op_to_node_container_pt_["jit_node"].at(idx);
  }

  void set_node_indices(std::vector<synNodeId> syn_node_ids) {
    for (auto syn_node : syn_node_ids) {
      op_to_node_container_pt_["jit_node"].emplace_back(syn_node);
    }
  }

  void clear_node_indices() {
    op_to_node_container_pt_["jit_node"].clear();
  }

  bool is_dynamic_graph() const {
    return dynamic_graph_;
  }

  void set_dynamic_graph(bool dynamic_graph = true) {
    dynamic_graph_ = dynamic_graph;
  }

  bool is_optim_output_sif_enabled() const {
    return enable_optim_output_sif_;
  }

  void set_optim_output_sif_enabled(bool optim_sif = true) {
    enable_optim_output_sif_ = optim_sif;
  }

  bool is_dry_run() const {
    return dry_run_;
  }

  uint32_t get_num_of_tensors() const {
    return numTensors;
  }

  void set_num_of_tensors(uint32_t num_tensors) {
    numTensors = num_tensors;
  }

  void set_num_of_const_tensors(uint32_t num_tensors) {
    numConstTensors = num_tensors;
  }

  void increment_const_tensors(uint32_t count = 1) {
    numConstTensors += count;
  }

  uint32_t get_num_of_const_tensors() {
    return numConstTensors;
  }

  void set_num_of_inter_tensors(uint32_t num_tensors) {
    numInterTensors = num_tensors;
  }

  uint32_t get_num_of_inter_tensors() {
    return numInterTensors;
  }

  void increment_shape_tensors(uint32_t count = 1) {
    numShapeTensors += count;
  }

  uint32_t get_num_of_shape_tensors() {
    return numShapeTensors;
  }

  uint32_t get_num_of_nodes() const {
    return numNodes;
  }

  void set_num_of_nodes(uint32_t num_nodes) {
    numNodes = num_nodes;
  }

  bool get_is_valid() const {
    return is_valid_;
  }

  void set_is_valid(bool flag) {
    is_valid_ = flag;
  }

  bool is_shape_agnostic_graph() const {
    return is_shape_agnostic_graph_;
  }

  void set_shape_agnostic_graph(bool shape_agnostic_graph = true) {
    is_shape_agnostic_graph_ = shape_agnostic_graph;
  }

  bool is_eager_mode() const {
    return eager_mode_;
  }

  const std::vector<synNodeId>& get_syn_node_id_vec() const {
    return syn_node_id_vec_;
  }

 private:
  using Op2NodeContainer =
      absl::flat_hash_map<std::string, absl::flat_hash_set<synNodeId>>;
  using Op2NodeContainerPt =
      absl::flat_hash_map<std::string, std::vector<synNodeId>>;
  using EdgeContainer =
      absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>;

  graph(device& device, std::string name);

  std::pair<std::vector<synTensorHandleMap>, std::vector<synNodeHandleMap>>
  duplicate(synGraphHandle& duplicate_graph_handle);

  void collect_dst_synapse_nodes(
      graph::Op2NodeContainer::mapped_type& dst_synapse_node_ids,
      const std::string& dst_node);
  void collect_dst_synapse_nodes(
      graph::Op2NodeContainer::mapped_type& dst_synapse_node_ids,
      const std::string& dst_node,
      absl::flat_hash_map<std::string, bool>& visited_nodes);
  synStatus set_synapse_control_edges();

  device& device_;
  const std::string name_;
  bool is_valid_{false};
  bool in_build_phase_{true};
  bool in_execution_phase_{false};
  bool graph_is_empty_{true};
  synGraphHandle graph_handle_{};
  Op2NodeContainer op_to_node_container_;
  Op2NodeContainerPt op_to_node_container_pt_;
  EdgeContainer control_edges_container_;
  EdgeContainer data_edges_container_;
  absl::optional<std::string> current_op_name_;
  bool dry_run_{false};
  bool dynamic_graph_{false};
  bool enable_optim_output_sif_{false};
  uint32_t numTensors = 0;
  uint32_t numConstTensors = 0;
  uint32_t numInterTensors = 0;
  uint32_t numShapeTensors = 0;
  uint32_t numNodes = 0;
  bool is_shape_agnostic_graph_{false};
  bool eager_mode_{false};
  std::vector<synNodeId> syn_node_id_vec_;
};

} // namespace synapse_helpers
