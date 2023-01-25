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

#include "jit_fixcolors.h"
#include <string>
#include <unordered_map>
#include "jit_fixcycles.h"
#include "union_find.h"

namespace habana {
namespace program {
namespace utils {

namespace {

std::unordered_set<std::string> merge_inputs = {
    "prim::ListUnpack",
    "aten::permute",
};
std::unordered_set<std::string> merge_outputs = {
    "prim::ListConstruct",
    "aten::permute",
};

} // namespace

void FixColors(
    SplittingDecision& decision,
    const torch::jit::Graph& graph,
    bool fix_cycles) {
  auto uf = UnionFind(1 + unsigned(decision.MaxColor()));

  for (auto* node : graph.nodes()) {
    std::string node_str = node->kind().toQualString();
    if (merge_inputs.count(node_str)) {
      // put inputs in the same partition
      for (auto input : node->inputs()) {
        auto input_node = input->node();
        if (input_node == graph.param_node())
          continue;
        uf.Merge(decision.colors.at(node), decision.colors.at(input_node));
      }
    }

    if (merge_outputs.count(node_str)) {
      // put outputs in the same partition
      for (auto output : node->outputs()) {
        for (auto user : output->uses()) {
          auto output_node = user.user;
          if (output_node == graph.return_node())
            continue;
          uf.Merge(decision.colors.at(node), decision.colors.at(output_node));
        }
      }
    }
  }

  uf.Substitute(decision);

  if (fix_cycles) {
    FixCycles(decision, uf, graph);
    uf.Substitute(decision);
  }
}

} // namespace utils
} // namespace program
} // namespace habana
