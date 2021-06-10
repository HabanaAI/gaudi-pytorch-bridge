/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_lazy/hpu_lazy_cache.h"
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
      %2.1 : Tensor = aten::mul(%0, %1)
      %2 : Tensor = aten::mul(%2.1, %1)
      %3 : Tensor = aten::add_(%2, %1, %12)
      %4 : Tensor = aten::mul(%2, %1)
      %5 : Tensor = aten::add(%2, %4, %12)
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

  torch::Tensor hx = x.to(torch::kHABANA);
  torch::Tensor hy = x.to(torch::kHABANA);
  auto inputs = habana_lazy_test::createStack({x, y});

  // std::cout << "PTI_DBG :: x : \n" << hx.to("cpu") << '\n';
  // std::cout << "PTI_DBG :: y : \n" << hy.to("cpu") << '\n';

  std::string id_str{"HabanaLaunchOp"};
  std::shared_ptr<habana::RecipeArgumentSpec> rargpsh1 =
      std::make_shared<habana::RecipeArgumentSpec>(jit_ir_graph, id_str);

  // std::cout << "PTI_DBG :: jit_ir_graph graph_hash_code : "
  //<< rargpsh1->graphHashCode() << '\n';
  EXPECT_EQ(rargpsh1->graphHashCode(), rargpsh1->hashCode());

  std::shared_ptr<habana::RecipeArgumentSpec> rargpsh2 =
      std::make_shared<habana::RecipeArgumentSpec>(
          false, inputs, jit_ir_graph, id_str);

  // std::cout << "PTI_DBG :: jit_ir_graph graph_hash_code : "
  //<< rargpsh2->graphHashCode()
  //<< ", offset_hash_code : " << rargpsh2->offsetHashCode()
  //<< ", hash_code : " << rargpsh2->hashCode() << '\n';

  EXPECT_EQ(rargpsh1->graphHashCode(), rargpsh2->graphHashCode());
  EXPECT_NE(rargpsh2->graphHashCode(), rargpsh2->hashCode());
}
