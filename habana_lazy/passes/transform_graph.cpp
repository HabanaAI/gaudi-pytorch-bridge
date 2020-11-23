/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "transform_graph.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include <nlohmann/json.hpp>
using json = nlohmannV340::json;

namespace habana_lazy {

using Graph = torch::jit::Graph;
using SubgraphRewriter = torch::jit::SubgraphRewriter;
using Pattern = std::tuple<std::string, std::string>;
using Patterns = std::vector<Pattern>;

std::string get_transform_graph_file() {
  return static_cast<std::string>(std::getenv("HABANA_TRANSFORM_GRAPH_FILE"));
}

Pattern make_pattern(const char* p, const char* r) {
  std::string s1 = p;
  std::string s2 = r;
  return make_tuple(s1, s2);
}

/**
 * Patterens are either defined in json file (for ex: refer in test/cpp folder
 * pattern json file has a dummy not an aten op define mmrelu etc..)
 * Or define some static patterns in this file.
 */
void get_patterns(Patterns& patterns) {
  // Get the Habana JSON file, which has replace patterns
  std::string tg_file = get_transform_graph_file();
  if (tg_file.empty()) {
    // No file specified, add here specific patterns to match
  } else {
    // Read JSON file and populate the data structure
    std::ifstream reader(tg_file);
    // auto j = json::parse(reader);
    json j;
    reader >> j;
    for (json::iterator it = j.begin(); it != j.end(); ++it) {
      auto k = it.value()["Pattern"];
      std::string p = R"()";
      for (auto lk : k) {
        p += lk;
        p += "\n";
      }
      std::string r = R"()";
      k = it.value()["ReplacePattern"];
      for (auto lk : k) {
        r += lk;
        r += "\n";
      }
      patterns.emplace_back(p, r);
    }
  }
}

void transform_graph(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter graph_rewriter;
  // Get all the patterns to be proccessed
  Patterns patterns;
  get_patterns(patterns);
  // Iterate thru each pattern and register for re writing
  for (auto& p : patterns) {
    graph_rewriter.RegisterRewritePattern(std::get<0>(p), std::get<1>(p));
  }
  // if there were patterns to be processed, then re-write the graph
  if (patterns.size()) {
    graph_rewriter.runOnGraph(graph);
  }
}

}; // namespace habana_lazy
