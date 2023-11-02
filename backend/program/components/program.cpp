/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "program.h"

namespace habana {
namespace program {

Cluster::Cluster(Cluster::Id id) : id_(id) {}

std::string Cluster::ToDebugString() const {
  std::string result;
  result += "Cluster id=";
  result += std::to_string(id_);
  result += "\n";
  for (std::size_t index = 0; index < outputs_.size(); ++index) {
    result += "- output ";
    result += std::to_string(index);
    result += " goes to";
    for (auto& p : outputs_[index]) {
      result += " ";
      result += p.toString();
    }
    result += "\n";
  }
  result += lazy_graph_->get_cached_graph()->toString();
  return result;
}

Cluster* GraphOfClusters::CreateCluster() {
  auto cluster = std::make_unique<Cluster>(freeNodeId++);
  auto cluster_ptr = cluster.get();
  nodes_[cluster->id_] = std::move(cluster);
  return cluster_ptr;
}

Cluster* GraphOfClusters::FindCluster(Cluster::Id id) const {
  if (id == SINK) {
    return nullptr;
  }
  return nodes_.at(id).get();
}

void GraphOfClusters::ExpandCluster(
    Cluster::Id expanded_cluster_id,
    std::unique_ptr<GraphOfClusters>&& expansion) {
  std::unordered_map<Cluster::Id, Cluster::Id> remapping;

  auto expanded_cluster = FindCluster(expanded_cluster_id);
  TORCH_CHECK_EQ(expanded_cluster_id, expanded_cluster->id_);

  auto createNewClusters = [&] {
    for (auto& p : expansion->nodes_) {
      auto old_id = p.first;
      auto& old_cluster = p.second;
      auto new_cluster = CreateCluster();
      remapping[old_id] = new_cluster->id_;
      new_cluster->lazy_graph_ = std::move(old_cluster->lazy_graph_);
      new_cluster->outputs_ = old_cluster->outputs_;
    }
    remapping[GraphOfClusters::SINK] = GraphOfClusters::SINK;
  };

  auto fixClusterOutputs = [&] {
    for (auto& p : remapping) {
      auto new_cluster_id = p.second;
      if (new_cluster_id == GraphOfClusters::SINK)
        continue;
      auto new_cluster = FindCluster(new_cluster_id);
      TORCH_CHECK_NOTNULL(new_cluster);
      for (auto& ports : new_cluster->outputs_) {
        for (auto& port : ports) {
          port.cluster = remapping.at(port.cluster);
        }
      }
    }
    for (auto& ports : expansion->inputs_) {
      for (auto& port : ports) {
        port.cluster = remapping.at(port.cluster);
      }
    }
  };

  auto collectPortsIndicesForCluster =
      [](Cluster::Id id, const PortVector& ps) -> std::vector<std::size_t> {
    std::vector<std::size_t> result;
    for (auto& p : ps) {
      if (p.cluster == id) {
        result.emplace_back(p.index);
      }
    }

    return result;
  };

  auto removeClusterFromPorts = [](Cluster::Id id, PortVector& ps) {
    auto pred = [id](const Port& p) { return p.cluster == id; };
    auto it = std::remove_if(ps.begin(), ps.end(), pred);
    ps.erase(it, ps.end());
  };

  auto fixSinkOutputs = [&] {
    for (auto& p : remapping) {
      auto new_cluster_id = p.second;
      if (new_cluster_id == GraphOfClusters::SINK)
        continue;
      auto new_cluster = FindCluster(new_cluster_id);
      TORCH_CHECK_NOTNULL(new_cluster);
      for (auto& ports : new_cluster->outputs_) {
        auto sinks =
            collectPortsIndicesForCluster(GraphOfClusters::SINK, ports);
        if (sinks.empty())
          continue;
        removeClusterFromPorts(GraphOfClusters::SINK, ports);

        for (auto output_index : sinks) {
          auto& outputs = expanded_cluster->outputs_.at(output_index);
          ports.insert(ports.end(), outputs.begin(), outputs.end());
        }
      }
    }
  };

  auto fixInputs = [&] {
    for (auto& p : nodes_) {
      auto& cluster = p.second;
      for (auto& ports : cluster->outputs_) {
        auto output_indices =
            collectPortsIndicesForCluster(expanded_cluster_id, ports);
        if (output_indices.empty())
          continue;
        removeClusterFromPorts(expanded_cluster_id, ports);

        for (auto output_index : output_indices) {
          auto& outputs = expansion->inputs_.at(output_index);
          ports.insert(ports.end(), outputs.begin(), outputs.end());
        }
      }
    }

    for (auto& ports : inputs_) {
      auto output_indices =
          collectPortsIndicesForCluster(expanded_cluster_id, ports);
      if (output_indices.empty())
        continue;
      removeClusterFromPorts(expanded_cluster_id, ports);

      for (auto output_index : output_indices) {
        auto& outputs = expansion->inputs_.at(output_index);
        ports.insert(ports.end(), outputs.begin(), outputs.end());
      }
    }
  };

  auto cleanup = [&] {
    nodes_.erase(expanded_cluster_id);
    expansion.reset();
  };

  createNewClusters();
  fixClusterOutputs();
  fixSinkOutputs();
  fixInputs();
  cleanup();
}

std::string GraphOfClusters::ToDebugString() const {
  std::string result;
  result += "Graph of clusters:\n";
  for (std::size_t index = 0; index < inputs_.size(); ++index) {
    result += "- input ";
    result += std::to_string(index);
    result += " goes to";
    for (auto& p : inputs_[index]) {
      result += " ";
      result += p.toString();
    }
    result += "\n";
  }

  for (auto& p : nodes_) {
    result += p.second->ToDebugString();
    result += "\n";
  }
  return result;
}

std::string GraphOfClusters::ToDebugSummary() const {
  std::string result;
  result += "Graph of clusters summary:\n";
  result += "Number of clusters: ";
  result += std::to_string(nodes_.size());
  result += "\n";
  result += "Sizes:";
  for (auto& p : nodes_) {
    result += " " +
        std::to_string(std::distance(
            p.second->lazy_graph_->get_cached_graph()->nodes().begin(),
            p.second->lazy_graph_->get_cached_graph()->nodes().end()));
  }
  result += "\n";
  return result;
}

ClusteredProgram::ClusteredProgram(
    std::unique_ptr<GraphOfClusters>&& graph_of_clusters)
    : graph_of_clusters_(std::move(graph_of_clusters)) {}

const ClusteredProgram::Schedule& ClusteredProgram::GetSchedule() const {
  return schedule_;
}

void ClusteredProgram::SetSchedule(Schedule&& schedule) {
  schedule_ = std::move(schedule);
}

GraphOfClusters* ClusteredProgram::GetGraphOfClusters() {
  return graph_of_clusters_.get();
}

std::string ClusteredProgram::ToDebugString() const {
  std::string result;
  result += "Clustered program this=";
  result += std::to_string((unsigned long)this);
  result += "\n";
  result += "Schedule: [";

  for (auto id : schedule_) {
    result += " ";
    result += std::to_string(id);
  }
  result += "]\n";
  result += graph_of_clusters_->ToDebugString();

  return result;
}

} // namespace program
} // namespace habana