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
#include "habana_lazy/hpu_lazy_cache.h"
#include "torch/csrc/jit/ir/ir.h"
namespace habana {
namespace program {

using LazyJitGraph = habana_lazy::OptimizedJITGraphAndMetaData;

/*
 * Cluster, represents part of computation.
 * Contains part of JIT IR Graph and unique id.
 */
struct Cluster {
  using Id = std::uint64_t;

  Cluster(Id id);

  Id id_;
  std::shared_ptr<LazyJitGraph> lazy_graph_;
};

using ClusterUPtr = std::unique_ptr<Cluster>;

/*
 * Graph of clusters.
 *
 * Simple graph structure for managing cluster and their interactions.
 */
class GraphOfClusters {
 public:
  Cluster* CreateCluster();

  Cluster* FindCluster(Cluster::Id id);

 private:
  std::unordered_map<Cluster::Id, ClusterUPtr> nodes_;
  Cluster::Id freeNodeId = 0;
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