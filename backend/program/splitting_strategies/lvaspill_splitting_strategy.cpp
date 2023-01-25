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
#include "lvaspill_splitting_strategy.h"
#include <deque>
#include "../utils/jit_bfstopsort.h"
#include "../utils/jit_liveness_analysis.h"

namespace habana {
namespace program {
namespace {

using Node = torch::jit::Node;
using Value = torch::jit::Value;
using Worklist = std::deque<const Node*>;

/*
 * Implementation of strategy itself
 */
class LvaSpillImpl {
 public:
  LvaSpillImpl(const LazyJitGraph& lazy_graph)
      : lazy_graph_(lazy_graph), graph_(*lazy_graph.get_cached_graph()) {}

  SplittingDecision Run() {
    ComputePessimisticSchedule();
    ComputeSpills();
    ColorPartitions();
    return std::move(decision_);
  }

 private:
  // We treat BFS based scheduling as pessimistic scenario
  void ComputePessimisticSchedule() {
    schedule_ = utils::BfsTopSort(graph_);
  }

  // Find nodes where number of bytes alive is bigger than threshold
  void ComputeSpills() {
    auto liveness = utils::AnalyseLiveness(lazy_graph_, schedule_);

    PT_BRIDGE_DEBUG(liveness.ToDebugString(schedule_));

    for (std::size_t i = 1; i < liveness.bytes_alive_before.size(); ++i) {
      auto bytes = liveness.bytes_alive_before[i];
      if (bytes > spill_threshold_) {
        spills_.push_back(schedule_[i]);
      }
    }
  }

  // Marks root and all direct/indicrect predecessors with given color.
  // Only uncolored nodes are marked.
  void ColorBackwardReachable(std::int64_t color, const Node* root) {
    Worklist worklist = {root};

    while (not worklist.empty()) {
      auto node = worklist.front();
      worklist.pop_front();

      auto it = decision_.colors.find(node);
      if (it != decision_.colors.end()) {
        // node is already colored, stopping here
        continue;
      }

      decision_.colors[node] = color;
      for (auto input_value : node->inputs()) {
        auto input_node = input_value->node();
        it = decision_.colors.find(input_node);
        if (it == decision_.colors.end()) {
          // predecessor not colored, adding to worklist
          worklist.push_back(input_node);
        }
      }
    }
  }

  /*
   * Create partitions
   */
  void ColorPartitions() {
    std::int64_t fresh_color = 0;

    // Use every spill point as root of new partition.
    for (auto spill : spills_) {
      ColorBackwardReachable(fresh_color++, spill);
    }

    // Create partition containing "tail".
    ColorBackwardReachable(fresh_color++, graph_.return_node());
  }

  const LazyJitGraph& lazy_graph_;
  const torch::jit::Graph& graph_;
  std::vector<const Node*> schedule_;
  std::vector<const Node*> spills_;
  SplittingDecision decision_;
  std::size_t spill_threshold_ = 1 * 1024 * 1024 * 1024;
};

} // namespace

SplittingDecision LvaSpillSplittingStrategy(const LazyJitGraph& lazy_graph) {
  return LvaSpillImpl(lazy_graph).Run();
}

} // namespace program
} // namespace habana
