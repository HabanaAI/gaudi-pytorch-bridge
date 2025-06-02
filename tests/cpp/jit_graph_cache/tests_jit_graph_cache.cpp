/**
 * Copyright (c) 2025 Intel Corporation
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
#include <gtest/gtest.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <memory>
#include <string_view>
#include "backend/jit_graph_cache.h"

namespace {
std::shared_ptr<torch::jit::Graph> Parse(const std::string& code) {
  auto graph = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(code, graph.get());
  return graph;
}

std::string createFakeCode(std::size_t lines = 78, std::size_t cols = 9) {
  const std::string line(cols, 'x');
  const std::size_t bytes_per_line = cols + 1;

  std::string code;
  code.reserve(lines * bytes_per_line);

  for (std::size_t i = 0; i < lines; ++i) {
    code.append(line).push_back('\n');
  }
  return code;
}
} // namespace

class JitGraphCacheTest : public ::testing::Test {};

TEST(JitGraphCacheTest, HashIgnoresIrComments) {
  const std::string IR_NO_COMMENTS =
      R"IR(graph(%0 : BFloat16(512, 32, strides=[32, 1], device=hpu:0),
  %1 : BFloat16(512, 1, strides=[1, 1], device=hpu:0),
  %2 : BFloat16(32, 1, strides=[1, 1], device=hpu:0)):
    %3 : int[] = prim::Constant[value=[32, 512]]()
    %4 : int[] = prim::Constant[value=[1, 32]]()
    %5 : int    = prim::Constant[value=0]()
    %6 : BFloat16(32, 512, strides=[512, 1], device=hpu:0) = aten::as_strided[deterministic=0](%0, %3, %4, %5)
    %7 : BFloat16(32, 1,  strides=[1, 1], device=hpu:0) = aten::mm[deterministic=0](%6, %1, %2)
    return (%7)
  )IR";

  auto g_no_comments = Parse(IR_NO_COMMENTS);
  ASSERT_NE(g_no_comments, nullptr);
  auto g_with_comments = Parse(IR_NO_COMMENTS);
  ASSERT_NE(g_with_comments, nullptr);

  torch::Tensor t0 = torch::empty({512, 32}, torch::dtype(torch::kBFloat16));
  torch::Tensor t1 = torch::empty({512, 1}, torch::dtype(torch::kBFloat16));
  torch::Tensor t2 = torch::empty({32, 1}, torch::dtype(torch::kBFloat16));

  std::vector<torch::jit::IValue> inputs_vec = {t0, t1, t2};
  at::ArrayRef<torch::jit::IValue> input_refs(inputs_vec);

  const std::string id = "test_graph";
  uint64_t unique_graph_cntr = 1;
  std::vector<bool> node_bcast_details;
  bool dynamic_graph = false;
  std::map<int64_t, std::vector<int64_t>> new_base_sizes;

  std::string fake_code = createFakeCode();
  auto src = std::make_shared<torch::jit::Source>(
      std::string_view(fake_code), std::string("<eval_with_key>"));

  torch::jit::SourceRange range(src, (77 * 10) + 5, 0);

  // set comments to each node
  for (auto node : g_with_comments->nodes()) {
    node->setSourceRange(range);
  }

  std::string op_strs1;
  size_t hash1 = 0;
  habana::ComputeGraphHashCode(
      g_no_comments,
      id,
      input_refs,
      op_strs1,
      hash1,
      unique_graph_cntr,
      node_bcast_details,
      dynamic_graph,
      new_base_sizes);

  std::string op_strs2;
  size_t hash2 = 0;
  habana::ComputeGraphHashCode(
      g_with_comments,
      id,
      input_refs,
      op_strs2,
      hash2,
      unique_graph_cntr,
      node_bcast_details,
      dynamic_graph,
      new_base_sizes);
  ASSERT_EQ(hash1, hash2);
}
