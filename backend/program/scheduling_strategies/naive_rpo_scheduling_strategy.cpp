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

#include "naive_rpo_scheduling_strategy.h"
#include <algorithm>

namespace habana {
namespace program {

SchedulingDecision NaiveRpoSchedulingStrategy(const GraphOfClusters& graph) {
  SchedulingDecision decision;

  std::unordered_set<Cluster::Id> visited;
  visited.insert(GraphOfClusters::SINK);

  std::function<void(Cluster::Id)> dfs = [&](Cluster::Id id) {
    if (visited.count(id))
      return;
    visited.insert(id);
    auto cluster = graph.FindCluster(id);

    for (auto& outputs : cluster->outputs_) {
      for (auto& output : outputs) {
        dfs(output.cluster);
      }
    }
    decision.scheduling.push_back(id);
  };

  for (auto& p : graph.nodes_) {
    dfs(p.first);
  }

  std::reverse(decision.scheduling.begin(), decision.scheduling.end());

  return decision;
} // namespace program

} // namespace program
} // namespace habana