/******************************************************************************
 * Copyright (C) 2020,2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/event.h"
#include "synapse_helpers/synapse_error.h"

namespace absl {
template <typename... Ts>
class variant;
} // namespace absl
namespace synapse_helpers {
class device;
class tensor;
} // namespace synapse_helpers

#define UNUSED __attribute__((unused))

namespace synapse_helpers {

class graph {
 public:
  graph() = delete;
  ~graph();
  graph(const graph&) = delete;
  graph& operator=(const graph&) = delete;
  graph(graph&&) noexcept;
  graph& operator=(graph&&) = delete;

  static synapse_error_v<graph> create(
      device& device,
      std::string name,
      bool dry_run = false);

  synapse_error_o add_node(
      std::vector<synTensor>&& inputs,
      std::vector<synTensor>&& outputs,
      void* const params,
      const unsigned params_size,
      const synapse_error_v<std::string>& node_type,
      synNodeId* ret_node_id = nullptr);

  template <typename ParamsT>
  synapse_error_o add_node(
      std::vector<synTensor>&& inputs,
      std::vector<synTensor>&& outputs,
      ParamsT* const params,
      const synapse_error_v<std::string>& node_type) {
    return add_node(
        std::move(inputs),
        std::move(outputs),
        params,
        sizeof(*params),
        node_type);
  }

  synStatus set_synapse_control_edges_pt(
      std::vector<synNodeId>,
      std::vector<synNodeId>);

  bool is_empty() const {
    return graph_is_empty_;
  }

  struct recipe_handle {
    synRecipeHandle syn_recipe_handle_{nullptr};
    std::string recipe_name_{""};
    bool graph_is_empty_{false};
    bool in_execution_phase_{false};
    uint64_t get_recipe_host_mem_size();

    explicit recipe_handle(){};
    ~recipe_handle();

    recipe_handle(const recipe_handle&) = delete;
    recipe_handle& operator=(const recipe_handle&) = delete;
    recipe_handle(recipe_handle&&) = delete;
    recipe_handle& operator=(recipe_handle&&) = delete;

   private:
    uint64_t recipe_size_ = 0;
  };

  synapse_error_v<std::shared_ptr<recipe_handle>> compile();

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

  static synapse_error_v<std::string> name_suffix_from_type(synDataType type);

  static synapse_error_v<uint64_t> query_workspace_size(
      const graph::recipe_handle& recipe_handle);

  static synapse_error_o launch(
      device& device,
      const graph::recipe_handle& recipe_handle,
      uint64_t workspace_size,
      std::vector<synLaunchTensorInfo>&& inputs_and_outputs_info,
      std::unique_ptr<device_ptr_lock>& address_lock,
      std::vector<shared_event>& ext_events);

  static synapse_error_o launch(
      device& device,
      const graph::recipe_handle& recipe_handle,
      uint64_t workspace_size,
      std::vector<synLaunchTensorInfo>& inputs_and_outputs_info,
      std::unique_ptr<device_ptr_lock>& address_lock,
      std::vector<shared_event>& ext_events);

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

  bool is_dry_run() const {
    return dry_run_;
  }

 private:
  using Op2NodeContainer =
      absl::flat_hash_map<std::string, absl::flat_hash_set<synNodeId>>;
  using Op2NodeContainerPt =
      absl::flat_hash_map<std::string, std::vector<synNodeId>>;
  using EdgeContainer =
      absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>;

  graph(device& device, std::string name);

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
  static std::mutex instance_lock_;
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
};

} // namespace synapse_helpers
