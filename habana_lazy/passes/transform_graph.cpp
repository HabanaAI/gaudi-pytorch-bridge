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

Patterns internal_patts = {
    // torch.all(tensor) pattern
    {"graph(%a):\n\
      %b : Tensor = aten::all(%a)\n\
      return (%b)",
     "graph(%x):\n\
      %11 : int = prim::Constant[value=11]()\n\
      %5 : None = prim::Constant()\n\
      %3 : bool = prim::Constant[value=0]()\n\
      %2 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to(%x, %2, %3, %3, %5)\n\
      %z : Tensor = aten::prod(%y, %5)\n\
      %a : Tensor = aten::to(%z, %11, %3, %3, %5)\n\
      return (%a)"},
    {"graph(%a):\n\
      %b : Tensor = aten::all[alpha=0](%a)\n\
      return (%b)",
     "graph(%x):\n\
      %11 : int = prim::Constant[value=11]()\n\
      %5 : None = prim::Constant()\n\
      %3 : bool = prim::Constant[value=0]()\n\
      %2 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to[alpha=0](%x, %2, %3, %3, %5)\n\
      %z : Tensor = aten::prod[alpha=0](%y, %5)\n\
      %a : Tensor = aten::to[alpha=0](%z, %11, %3, %3, %5)\n\
      return (%a)"},
    {"graph(%a):\n\
      %b : Tensor = aten::all[alpha=1](%a)\n\
      return (%b)",
     "graph(%x):\n\
      %11 : int = prim::Constant[value=11]()\n\
      %5 : None = prim::Constant()\n\
      %3 : bool = prim::Constant[value=0]()\n\
      %2 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to[alpha=1](%x, %2, %3, %3, %5)\n\
      %z : Tensor = aten::prod[alpha=1](%y, %5)\n\
      %a : Tensor = aten::to[alpha=1](%z, %11, %3, %3, %5)\n\
      return (%a)"},
    // torch.all(tensor, dim, keepdim) pattern
    {"graph(%a, %dim : int, %keepdim : bool):\n\
      %b : Tensor = aten::all(%a, %dim, %keepdim)\n\
      return (%b)",
     "graph(%x, %dim : int, %keepdim : bool):\n\
      %16 : int = prim::Constant[value=11]()\n\
      %7 : None = prim::Constant()\n\
      %5 : bool = prim::Constant[value=0]()\n\
      %4 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to(%x, %4, %5, %5, %7)\n\
      %z : Tensor = hpu::prod_dim_Int(%y, %dim, %keepdim, %7)\n\
      %a : Tensor = aten::to(%z, %16, %5, %5, %7)\n\
      return (%a)"},
    {"graph(%a, %dim : int, %keepdim : bool):\n\
      %b : Tensor = aten::all[alpha=0](%a, %dim, %keepdim)\n\
      return (%b)",
     "graph(%x, %dim : int, %keepdim : bool):\n\
      %16 : int = prim::Constant[value=11]()\n\
      %7 : None = prim::Constant()\n\
      %5 : bool = prim::Constant[value=0]()\n\
      %4 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to[alpha=0](%x, %4, %5, %5, %7)\n\
      %z : Tensor = hpu::prod_dim_Int[alpha=0](%y, %dim, %keepdim, %7)\n\
      %a : Tensor = aten::to[alpha=0](%z, %16, %5, %5, %7)\n\
      return (%a)"},
    {"graph(%a, %dim : int, %keepdim : bool):\n\
      %b : Tensor = aten::all[alpha=1](%a, %dim, %keepdim)\n\
      return (%b)",
     "graph(%x, %dim : int, %keepdim : bool):\n\
      %16 : int = prim::Constant[value=11]()\n\
      %7 : None = prim::Constant()\n\
      %5 : bool = prim::Constant[value=0]()\n\
      %4 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to[alpha=1](%x, %4, %5, %5, %7)\n\
      %z : Tensor = hpu::prod_dim_Int[alpha=1](%y, %dim, %keepdim, %7)\n\
      %a : Tensor = aten::to[alpha=1](%z, %16, %5, %5, %7)\n\
      return (%a)"},
    // torch.all(tensor, dim, keepdim) pattern for Lazy
    {"graph(%a, %dim : int, %keepdim : bool):\n\
      %b : Tensor = hpu::all_dim(%a, %dim, %keepdim)\n\
      return (%b)",
     "graph(%x, %dim : int, %keepdim : bool):\n\
      %16 : int = prim::Constant[value=11]()\n\
      %7 : None = prim::Constant()\n\
      %5 : bool = prim::Constant[value=0]()\n\
      %4 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to(%x, %4, %5, %5, %7)\n\
      %z : Tensor = hpu::prod_dim_Int(%y, %dim, %keepdim, %7)\n\
      %a : Tensor = aten::to(%z, %16, %5, %5, %7)\n\
      return (%a)"},
    {"graph(%a, %dim : int, %keepdim : bool):\n\
      %b : Tensor = hpu::all_dim[alpha=0](%a, %dim, %keepdim)\n\
      return (%b)",
     "graph(%x, %dim : int, %keepdim : bool):\n\
      %16 : int = prim::Constant[value=11]()\n\
      %7 : None = prim::Constant()\n\
      %5 : bool = prim::Constant[value=0]()\n\
      %4 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to[alpha=0](%x, %4, %5, %5, %7)\n\
      %z : Tensor = hpu::prod_dim_Int[alpha=0](%y, %dim, %keepdim, %7)\n\
      %a : Tensor = aten::to[alpha=0](%z, %16, %5, %5, %7)\n\
      return (%a)"},
    {"graph(%a, %dim : int, %keepdim : bool):\n\
      %b : Tensor = hpu::all_dim[alpha=1](%a, %dim, %keepdim)\n\
      return (%b)",
     "graph(%x, %dim : int, %keepdim : bool):\n\
      %16 : int = prim::Constant[value=11]()\n\
      %7 : None = prim::Constant()\n\
      %5 : bool = prim::Constant[value=0]()\n\
      %4 : int = prim::Constant[value=6]()\n\
      %y : Tensor = aten::to[alpha=1](%x, %4, %5, %5, %7)\n\
      %z : Tensor = hpu::prod_dim_Int[alpha=1](%y, %dim, %keepdim, %7)\n\
      %a : Tensor = aten::to[alpha=1](%z, %16, %5, %5, %7)\n\
      return (%a)"}};

std::string get_transform_graph_file() {
  if (std::getenv("HABANA_TRANSFORM_GRAPH_FILE")) {
    return static_cast<std::string>(std::getenv("HABANA_TRANSFORM_GRAPH_FILE"));
  } else {
    return {};
  }
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
