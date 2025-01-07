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

#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "habana_lazy/passes/replace_inplace_ops.h"

using namespace torch::jit;

struct ReplaceInplaceOpsPassTests : public ::testing::Test {
  std::shared_ptr<Graph> Parse(const std::string& code) {
    auto graph = std::make_shared<Graph>();
    parseIR(code, graph.get());
    return graph;
  }

  std::string Print(std::shared_ptr<Graph> graph) {
    std::stringstream ss;
    graph->print(ss);
    return ss.str();
  }

  std::size_t Count(const std::string& str, const std::string& substr) {
    std::size_t pos = 0;
    std::size_t count = 0;
    while (pos != std::string::npos) {
      pos = str.find(substr, pos);
      if (pos != std::string::npos) {
        count += 1;
        pos += substr.size();
      }
    }

    return count;
  }
};

/* Simple test for quick fix inside replace_inplace_ops graph pass.
 * Before fix the graph pass was not detecting aten::zero_ as inplace op
 * and was replacing proceeding inplace instruction. Here we have real-world
 * scenario JIT IR dumped from AdamW optimizer. We assert preservation of
 * inplace multiplications after zero_ operation.
 */
TEST_F(ReplaceInplaceOpsPassTests, preserving_inplace_mul_) {
  const char* INPUT_JIR_IR = R"IR(
graph(%id_53_hpu__input : Float(device=hpu:0),
      %id_56_hpu__input : Float(device=hpu:0),
      %id_48_hpu__input : Float(device=hpu:0),
      %id_12_hpu__input : Float(1, 33554432, strides=[33554432, 1], device=hpu:0),
      %id_39_hpu__input : Float(device=hpu:0),
      %id_33_hpu__input : Float(device=hpu:0),
      %id_22_hpu__input : Float(1, 33554432, strides=[33554432, 1], device=hpu:0),
      %id_36_hpu__input : Float(device=hpu:0),
      %id_30_hpu__input : Float(device=hpu:0),
      %id_17_hpu__input : Float(1, 33554432, strides=[33554432, 1], device=hpu:0),
      %id_8_hpu__input : Float(1, 33554432, strides=[33554432, 1], device=hpu:0),
      %id_27_hpu__input : Float(device=hpu:0)):
  %22 : float = prim::Constant[value=1.]()
  %13 : int = prim::Constant[value=6]()
  %12 : int = prim::Constant[value=1]()
  %id_57_hpu__cast : Float(device=hpu:0) = hpu::cast[alpha=0](%id_56_hpu__input, %13)
  %id_22_hpu__control_edge_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = hpu::control_edge_[alpha=0](%id_22_hpu__input)
  %id_22_aten__zero_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = aten::zero_[alpha=0](%id_22_hpu__control_edge_)
  %id_22_aten__mul_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = aten::mul_[alpha=0](%id_22_aten__zero_, %id_33_hpu__input)
  %id_22_hpu__addcmul_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = hpu::addcmul_[alpha=0](%id_22_aten__mul_, %id_12_hpu__input, %id_12_hpu__input, %id_39_hpu__input)
  %id_42_aten__sqrt : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = aten::sqrt[alpha=0](%id_22_hpu__addcmul_)
  %id_49_aten__div : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = aten::div[alpha=0](%id_42_aten__sqrt, %id_48_hpu__input)
  %id_49_hpu__add_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = hpu::add[alpha=0](%id_49_aten__div, %id_57_hpu__cast, %12)
  %id_40_aten__mul : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = aten::mul[alpha=0](%id_12_hpu__input, %id_36_hpu__input)
  %id_17_hpu__control_edge_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = hpu::control_edge_[alpha=0](%id_17_hpu__input)
  %id_17_aten__zero_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = aten::zero_[alpha=0](%id_17_hpu__control_edge_)
  %id_17_aten__mul_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = aten::mul_[alpha=0](%id_17_aten__zero_, %id_30_hpu__input)
  %id_17_hpu__add_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = hpu::add_[alpha=0](%id_17_aten__mul_, %id_40_aten__mul, %22)
  %id_8_aten__mul_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = aten::mul_[alpha=0](%id_8_hpu__input, %id_27_hpu__input)
  %id_8_hpu__addcdiv_ : Float(1, 33554432, strides=[33554432, 1], device=hpu:0) = hpu::addcdiv_[alpha=0](%id_8_aten__mul_, %id_17_hpu__add_, %id_49_hpu__add_, %id_53_hpu__input)
  return (%id_8_hpu__addcdiv_, %id_17_hpu__add_, %id_22_hpu__addcmul_)
)IR";

  auto graph = Parse(INPUT_JIR_IR);
  ASSERT_NE(graph, nullptr);

  habana_lazy::replace_inplace_ops(graph);

  // Simple test, we print graph to string and check number of occurences of
  // substring denoting desired operation.

  auto printed_graph = Print(graph);

  const std::size_t expected_number_of_inplace_muls = 3;

  ASSERT_EQ(
      expected_number_of_inplace_muls, Count(printed_graph, "aten::mul_"));
}
