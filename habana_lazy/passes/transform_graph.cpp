/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "transform_graph.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include <nlohmann/json.hpp>

#include <fstream>

using json = nlohmannV340::json;

namespace habana_lazy {
using Graph = torch::jit::Graph;
using SubgraphRewriter = torch::jit::SubgraphRewriter;
using Pattern = std::tuple<std::string, std::string>;
using Patterns = std::vector<Pattern>;

Patterns internal_patts = {};

std::string get_transform_graph_file() {
  if (std::getenv("HABANA_TRANSFORM_GRAPH_FILE")) {
    return static_cast<std::string>(std::getenv("HABANA_TRANSFORM_GRAPH_FILE"));
  } else {
    return {};
  }
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

  // add internal patterns written to realize complex OPs
  // using existing simple OPs
  for (unsigned int i = 0; i < internal_patts.size(); i++) {
    patterns.emplace_back(internal_patts.at(i));
  }
}

void transform_graph(std::shared_ptr<Graph>& graph) {
  // Get all the patterns to be proccessed
  Patterns patterns;
  get_patterns(patterns);
  // Iterate thru each pattern and register for re writing
  // if there were patterns to be processed, then re-write the graph
  for (auto& p : patterns) {
    SubgraphRewriter graph_rewriter;
    graph_rewriter.RegisterRewritePattern(std::get<0>(p), std::get<1>(p));
    graph_rewriter.runOnGraph(graph);
  }
}

}; // namespace habana_lazy
