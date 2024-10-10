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

#include "jit_bfstopsort.h"

namespace habana {
namespace program {
namespace utils {

namespace {

using Node = torch::jit::Node;
using Value = torch::jit::Value;
using Worklist = std::deque<const Node*>;

/*
 * BFS-based topological sorting.
 *
 * Just implementing Kahn's algorithm. For each node we maintain
 * counter of unvisited predecessors. When node is visited we
 * decrease counter of its users and when some counter reaches zero
 * then we add user it to the queue.
 */
class BfsTopsortImpl {
 public:
  BfsTopsortImpl(const torch::jit::Graph& graph) : graph_(graph) {}

  std::vector<const Node*> Run() {
    PrepareCounters();
    FindRoots();
    Sort();
    return std::move(schedule_);
  }

 private:
  using Counters = std::unordered_map<const Node*, std::int64_t>;

  void PrepareCounters() {
    for (auto node : graph_.nodes()) {
      counters_[node] = node->inputs().size();
    }
    counters_[graph_.return_node()] = graph_.return_node()->inputs().size();
  }

  void DecrementSuccessors(const Node* producer) {
    for (auto output_value : producer->outputs()) {
      for (auto use : output_value->uses()) {
        counters_[use.user] -= 1;
        TORCH_CHECK_GE(counters_[use.user], 0);
        if (counters_[use.user] == 0) {
          worklist_.push_back(use.user);
        }
      }
    }
  }

  void FindRoots() {
    for (auto& p : counters_) {
      if (p.second == 0) {
        worklist_.push_back(p.first);
      }
    }

    // Unblock users of formal parameters
    DecrementSuccessors(graph_.param_node());
  }

  void Sort() {
    while (not worklist_.empty()) {
      auto node = worklist_.front();
      worklist_.pop_front();
      if (node == graph_.return_node()) {
        continue;
      }
      schedule_.push_back(node);
      DecrementSuccessors(node);
    }
  }

  std::vector<const Node*> schedule_;
  const torch::jit::Graph& graph_;
  Counters counters_;
  Worklist worklist_;
};

} // namespace

std::vector<const torch::jit::Node*> BfsTopSort(
    const torch::jit::Graph& graph) {
  return BfsTopsortImpl(graph).Run();
}

} // namespace utils
} // namespace program
} // namespace habana
