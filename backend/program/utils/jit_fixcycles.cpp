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

#include "jit_fixcycles.h"

namespace habana {
namespace program {
namespace utils {
namespace {

/*
 * Implementation of `FixCycles`. It is slow implementation, but we can improve
 * it later. It may be called only during partitioning.
 *
 * Algorithm will traverse graph and for each node will check if we can reach a
 * different node with the same color.
 */
struct FixCyclesImpl {
  FixCyclesImpl(
      SplittingDecision& decision,
      UnionFind& uf,
      const torch::jit::Graph& graph)
      : decision_(decision), uf_(uf), graph_(graph) {}

  void Run() {
    for (auto node : graph_.nodes()) {
      RunNode(node);
    }
  }

  // Helper usable in case someone needs to add debug printfs
  [[maybe_unused]] std::string C(const torch::jit::Node* node) {
    auto node_color = decision_.colors.at(node);
    std::string ret;
    ret += "node_color=";
    ret += std::to_string(node_color);
    ret += "[";
    ret += std::to_string(uf_.Find(node_color));
    ret += "]";
    return ret;
  }

  void RunNode(const torch::jit::Node* node) {
    auto node_color = decision_.colors.at(node);
    for (auto output_value : node->outputs()) {
      for (auto use : output_value->uses()) {
        if (use.user == graph_.return_node())
          continue;
        CheckChild(use.user, node_color);
      }
    }
  }

  bool CheckChild(const torch::jit::Node* node, std::int64_t source_color) {
    auto node_color = decision_.colors.at(node);
    if (uf_.Eq(node_color, source_color)) {
      return true;
    }

    bool result = false;
    for (auto output_value : node->outputs()) {
      for (auto use : output_value->uses()) {
        if (use.user != graph_.return_node()) {
          result |= CheckChild(use.user, source_color);
        }
      }
    }

    if (result) {
      uf_.Merge(node_color, source_color);
    }

    return result;
  }

  SplittingDecision& decision_;
  UnionFind& uf_;
  const torch::jit::Graph& graph_;
};

} // namespace

void FixCycles(
    SplittingDecision& decision,
    UnionFind& uf,
    const torch::jit::Graph& graph) {
  FixCyclesImpl(decision, uf, graph).Run();
}

} // namespace utils
} // namespace program
} // namespace habana
