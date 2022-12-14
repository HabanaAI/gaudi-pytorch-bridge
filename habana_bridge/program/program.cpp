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

#include "program.h"

namespace habana {
namespace program {

Cluster::Cluster(Cluster::Id id) : id_(id) {}

Cluster* GraphOfClusters::CreateCluster() {
  auto cluster = std::make_unique<Cluster>(freeNodeId++);
  auto cluster_ptr = cluster.get();
  nodes_[cluster->id_] = std::move(cluster);
  return cluster_ptr;
}

Cluster* GraphOfClusters::FindCluster(Cluster::Id id) {
  return nodes_.at(id).get();
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

} // namespace program
} // namespace habana