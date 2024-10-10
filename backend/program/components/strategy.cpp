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

#include "strategy.h"
#include "../scheduling_strategies/naive_rpo_scheduling_strategy.h"
#include "../splitting_strategies/eager_splitting_strategy.h"
#include "../splitting_strategies/lvaspill_splitting_strategy.h"

namespace habana {
namespace program {

SplittingDecision::operator bool() const {
  return not colors.empty();
}

std::int64_t SplittingDecision::MaxColor() const {
  std::int64_t color = 0;
  for (auto& p : colors) {
    color = std::max(color, p.second);
  }
  return color;
}

namespace {

struct SplittingDecisionValidatorImpl {
  SplittingDecisionValidatorImpl(
      const SplittingDecision& decision,
      const torch::jit::Graph& graph)
      : decision_(decision), graph_(graph) {}

  bool Run() {
    if (not ValidateAllNodesAreColored())
      return false;
    if (not ValidateNoCycles())
      return false;
    return true;
  }

  bool ValidateAllNodesAreColored() {
    for (auto node : graph_.nodes()) {
      auto it = decision_.colors.find(node);
      // Check if node has color
      if (it == decision_.colors.end()) {
        return false;
      }
      // Check if color is valid
      if (it->second < 0)
        return false;
    }

    return true;
  }

  bool ValidateNoCycles() {
    auto colors = decision_.colors;

    // Extend coloring to special nodes
    for (auto input_value : graph_.inputs()) {
      colors[input_value->node()] = -1;
    }
    colors[graph_.return_node()] = -2;

    // Build graph of colors
    std::unordered_map<std::int64_t, std::unordered_set<std::int64_t>>
        successors;

    for (auto node : graph_.nodes()) {
      auto node_color = colors.at(node);
      successors[node_color];
      for (auto input_value : node->inputs()) {
        auto input_node = input_value->node();
        auto input_color = colors.at(input_node);
        successors[input_color];
        if (input_color != node_color) {
          successors[input_color].insert(node_color);
        }
      }
    }

    // Check for cycle using DFS
    std::unordered_map<std::int64_t, bool> visited;

    // color not in visited => not visited
    // visited[color] == false => visited, but not processed
    // visited[color] == true => visited and fully processed

    std::function<bool(std::int64_t)> dfs = [&](std::int64_t color) {
      auto it = visited.find(color);
      if (it != visited.end()) {
        return it->second;
      }

      visited[color] = false;

      for (auto succ : successors.at(color)) {
        if (not dfs(succ))
          return false;
      }

      visited[color] = true;

      return true;
    };

    for (auto& p : successors) {
      if (not dfs(p.first))
        return false;
    }

    return true;
  }

  const SplittingDecision& decision_;
  const torch::jit::Graph& graph_;
};

} // namespace

bool SplittingDecision::Validate(const LazyJitGraph& lazy_graph) const {
  auto algo =
      SplittingDecisionValidatorImpl(*this, *lazy_graph.get_cached_graph());
  return algo.Run();
}

SchedulingDecision::operator bool() const {
  return not scheduling.empty();
}

bool SchedulingDecision::Validate(const GraphOfClusters& graph) const {
  (void)graph;
  // Not yet implemented
  return true;
}

namespace {

const std::string DEFAULT_SPLIT_STR = "eager";
const std::string DEFAULT_SCHED_STR = "naive";

std::unordered_map<std::string, SplittingStrategy> splitting_strategies = {
    {"eager", EagerSplittingStrategy},
    {"lvaspill", LvaSpillSplittingStrategy},
};

std::unordered_map<std::string, SchedulingStrategy> scheduling_strategies = {
    {"naive", NaiveRpoSchedulingStrategy}};

} // namespace

SplittingStrategy GetSplittingStrategy() {
  std::string name = GET_ENV_FLAG_NEW(PT_HPU_CLUSTERED_PROGRAM_SPLIT_STR);
  if (name == "default") {
    name = DEFAULT_SPLIT_STR;
  }
  auto it = splitting_strategies.find(name);
  if (it == splitting_strategies.end()) {
    PT_BRIDGE_WARN(
        "Unknown splitting strategy: ", name, " -- falling back to eager");
    return EagerSplittingStrategy;
  }
  return it->second;
}

SchedulingStrategy GetSchedulingStrategy() {
  std::string name = GET_ENV_FLAG_NEW(PT_HPU_CLUSTERED_PROGRAM_SCHED_STR);
  if (name == "default") {
    name = DEFAULT_SCHED_STR;
  }
  auto it = scheduling_strategies.find(name);
  if (it == scheduling_strategies.end()) {
    PT_BRIDGE_WARN(
        "Unknown scheduling strategy: ", name, " -- falling back to naive");
    return NaiveRpoSchedulingStrategy;
  }
  return it->second;
}

} // namespace program
} // namespace habana