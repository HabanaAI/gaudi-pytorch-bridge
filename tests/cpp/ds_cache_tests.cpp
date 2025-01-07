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

#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#include "backend/jit_graph_cache.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "habana_helpers/logging.h"
#include "habana_lazy_test_infra.h"

TEST(DS_CacheTest, UniqueTokenGenTest) {
  auto t0 = habana_helpers::UniqueTokenGenerator::get_gen().token();
  auto tinit = habana_helpers::Bucket::uninitialized_token;
  EXPECT_GT(t0, tinit);
  std::unordered_set<uint64_t> S;
  S.insert(t0);

  for (int i = 0; i < 10000; i++) {
    auto tok = habana_helpers::UniqueTokenGenerator::get_gen().token();
    EXPECT_EQ(S.count(tok), 0);
    S.insert(tok);
  }
}

TEST(DS_CacheTest, JIT_IR_GraphKeyTest) {
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %12 : int = prim::Constant[value=1]()
      %2.1 : Tensor = aten::mul[deterministic=0](%0, %1)
      %2 : Tensor = aten::mul[deterministic=0](%2.1, %1)
      %3 : Tensor = aten::add_[deterministic=0](%2, %1, %12)
      %4 : Tensor = aten::mul[deterministic=0](%2, %1)
      %5 : Tensor = aten::add[deterministic=0](%2, %4, %12)
      return (%5))IR";

  auto jit_ir_graph = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_string, jit_ir_graph.get());

  jit_ir_graph->lint();
  // std::cout << "PTF_DBG :: "
  //<< "Testing JIT IR Gprah creation" << '\n';
  // std::cout << "PTF_DBG :: " << __FUNCTION__ << " : "
  //<< "JIT IR graph" << '\n'
  //<< "----" << '\n'
  //<< jit_ir_graph->toString() << "----" << '\n';

  torch::Tensor x = torch::randn({5, 5}, torch::requires_grad());
  torch::Tensor y = torch::randn({5, 5}, torch::requires_grad());

  torch::Tensor hx = x.to(torch::kHPU);
  torch::Tensor hy = x.to(torch::kHPU);
  auto inputs = habana_lazy_test::createStack({hx, hy});

  std::string id_str{"HabanaLaunchOp"};
  size_t graphKey = 0;
  std::string op_strs = std::string();
  habana::ComputeGraphHashCode(jit_ir_graph, id_str, inputs, op_strs, graphKey);
  std::shared_ptr<habana::RecipeArgumentSpec> rargpsh1 =
      std::make_shared<habana::RecipeArgumentSpec>(inputs, graphKey, op_strs);

  EXPECT_EQ(rargpsh1->graphWithPermuteHashCode(), rargpsh1->hashCode());

  size_t sym_hash_code = habana::ComputeSymSizeHashCode(inputs);
  size_t perm_hash_code = habana::ComputePermutationHashCode(inputs);

  std::shared_ptr<habana::RecipeArgumentSpec> rargpsh2 =
      std::make_shared<habana::RecipeArgumentSpec>(
          false,
          inputs,
          jit_ir_graph,
          graphKey,
          op_strs,
          sym_hash_code,
          perm_hash_code);

  EXPECT_EQ(rargpsh1->graphHashCode(), rargpsh2->graphHashCode());
  EXPECT_NE(rargpsh2->graphHashCode(), rargpsh2->hashCode());
}

TEST(DS_CacheTest, ArgumentSpec) {
  torch::Tensor x = torch::randn({5, 5}, torch::requires_grad());
  torch::Tensor y = torch::randn({5, 5}, torch::requires_grad());

  torch::Tensor hx = x.to(torch::kHPU);
  torch::Tensor hy = x.to(torch::kHPU);
  auto inputs = habana_lazy_test::createStack({hx, hy});

  std::cout.setf(std::ios::unitbuf);
  std::string jit_instr{"a, b"};
  std::string jit_grstr =
      "def fn(" + jit_instr + "):  return " + jit_instr + "\n";
  std::shared_ptr<torch::jit::Graph> gr =
      toGraphFunction(torch::jit::compile(jit_grstr)->get_function("fn"))
          .graph();

  torch::jit::ArgumentSpecCreator as_creator(*gr);
  bool with_grad{true};
  torch::jit::ArgumentSpec as_regular = as_creator.create(with_grad, inputs);

  torch::jit::ArgumentSpec as_direct(2, 0);
  as_direct.addTensor(inputs[0], with_grad);
  as_direct.addTensor(inputs[1], with_grad);

  PT_TEST_DEBUG(
      "PTI_DBG :: ",
      " as_regular.hashCode()=",
      as_regular.hashCode(),
      " as_direct.hashCode()=",
      as_direct.hashCode());

  EXPECT_EQ(as_regular.hashCode(), as_direct.hashCode());
}
