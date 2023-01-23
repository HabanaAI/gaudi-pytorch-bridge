/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#pragma once
#include <memory>
#include <vector>
#include "backend/jit_graph_cache.h"
#include "torch/csrc/jit/ir/ir.h"
namespace habana {
namespace program {

using LazyJitGraph = habana::OptimizedJITGraphAndMetaData;

struct Port {
  Port() = default;
  Port(std::int64_t _cluster, std::size_t _index)
      : cluster(_cluster), index(_index) {}

  bool operator==(const Port& other) const {
    return std::tie(cluster, index) == std::tie(other.cluster, other.index);
  }

  bool operator<(const Port& other) const {
    return std::tie(cluster, index) < std::tie(other.cluster, other.index);
  }

  std::string toString() const {
    std::string res;
    res += "Port(cluster=";
    res += std::to_string(cluster);
    res += ", port=";
    res += std::to_string(index);
    res += ")";
    return res;
  }

  std::int64_t cluster = 0;
  std::size_t index = 0;
};

using PortVector = std::vector<Port>;

/*
 * Cluster, represents part of computation.
 * Contains part of JIT IR Graph and unique id.
 */
struct Cluster {
  using Id = std::int64_t;

  Cluster(Id id);

  Id id_;
  std::shared_ptr<LazyJitGraph> lazy_graph_;

  std::vector<PortVector> outputs_;
};

using ClusterUPtr = std::unique_ptr<Cluster>;

/*
 * Graph of clusters.
 *
 * Simple graph structure for managing cluster and their interactions.
 */
class GraphOfClusters {
 public:
  static constexpr Cluster::Id SINK = 0;

  Cluster* CreateCluster();

  Cluster* FindCluster(Cluster::Id id) const;
  Cluster* ExpandCluster(
      Cluster::Id id,
      std::unique_ptr<GraphOfClusters>&& expansion);

  std::vector<PortVector> inputs_;
  std::unordered_map<Cluster::Id, ClusterUPtr> nodes_;

 private:
  Cluster::Id freeNodeId = 1;
};

/*
 * Clustered program.
 *
 * Contains graph of clusters representing computation and selected schedule.
 *
 * TODO: should it contains mutex?
 */
class ClusteredProgram {
 public:
  using Schedule = std::vector<Cluster::Id>;

  ClusteredProgram(std::unique_ptr<GraphOfClusters>&& graph_of_clusters);

  const Schedule& GetSchedule() const;

  void SetSchedule(Schedule&& schedule);

  GraphOfClusters* GetGraphOfClusters();

 private:
  std::unique_ptr<GraphOfClusters> graph_of_clusters_;
  std::vector<Cluster::Id> schedule_;
};

using ClusteredProgramSPtr = std::shared_ptr<ClusteredProgram>;

} // namespace program
} // namespace habana
