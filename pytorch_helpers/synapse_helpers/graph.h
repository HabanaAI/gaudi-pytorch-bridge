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

  static synapse_error_v<graph> create(device& device, std::string name);

  synapse_error_o add_node(
      std::vector<synTensor>&& inputs,
      std::vector<synTensor>&& outputs,
      void* const params,
      const unsigned params_size,
      std::string&& node_type);

  template <typename ParamsT>
  synapse_error_o add_node(
      std::vector<synTensor>&& inputs,
      std::vector<synTensor>&& outputs,
      ParamsT* const params,
      std::string&& node_type) {
    return add_node(
        std::move(inputs),
        std::move(outputs),
        params,
        sizeof(*params),
        std::move(node_type));
  }

  bool is_empty() const {
    return graph_is_empty_;
  }

  struct recipe_handle {
    synRecipeHandle syn_recipe_handle_{nullptr};
    std::string recipe_name_{""};
    bool graph_is_empty_{false};
    bool in_execution_phase_{false};
    device& device_;

    explicit recipe_handle(device& device) : device_{device} {};
    ~recipe_handle();

    recipe_handle(const recipe_handle&) = delete;
    recipe_handle& operator=(const recipe_handle&) = delete;
    recipe_handle(recipe_handle&&) = delete;
    recipe_handle& operator=(recipe_handle&&) = delete;
  };

  synapse_error_v<std::shared_ptr<recipe_handle>> compile();

  struct launch_info {
    explicit launch_info(device& device) : device_{device} {}
    launch_info(const launch_info&) = delete;
    launch_info& operator=(const launch_info&) = delete;
    launch_info(launch_info&&) = delete;
    launch_info& operator=(launch_info&&) = delete;
    ~launch_info() = default;

    device& device_;

    uint64_t workspace_buffer_size_{0};

    std::chrono::steady_clock::time_point runtime_measure_start_{};
    std::vector<std::reference_wrapper<tensor>> outputs_{};

   private:
    std::string recipe_name_{""};
  };

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

  static synapse_error_o create_launch_info(
      launch_info& handle,
      const graph::recipe_handle& recipe_handle);

  static synapse_error_o launch(
      launch_info& handle,
      const graph::recipe_handle& recipe_handle,
      const std::vector<synLaunchTensorInfoDSD>& inputs_and_outputs_info);

  const std::string& name() const {
    return name_;
  }
  synGraphHandle get_graph_handle() const {
    return *graph_handle_;
  }

  std::vector<std::string> get_nodes() {
    return nodeType_;
  }

 private:
  using Op2NodeContainer =
      absl::flat_hash_map<std::string, absl::flat_hash_set<synNodeId>>;
  using EdgeContainer =
      absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>;
  void collect_dst_synapse_nodes(
      graph::Op2NodeContainer::mapped_type& dst_synapse_node_ids,
      const std::string& dst_node);
  synStatus set_synapse_control_edges();
  graph(device& device, std::string name);

  device& device_;
  const std::string name_;
  bool is_valid_;
  static std::mutex instance_lock_;
  bool in_build_phase_;
  bool in_execution_phase_;
  bool graph_is_empty_{true};
  std::unique_ptr<synGraphHandle> graph_handle_;
  Op2NodeContainer op_to_node_container_;
  EdgeContainer control_edges_container_;
  EdgeContainer data_edges_container_;
  absl::optional<std::string> current_op_name_;
  std::vector<std::string> nodeType_{};
};

} // namespace synapse_helpers
