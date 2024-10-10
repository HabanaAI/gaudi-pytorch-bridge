/******************************************************************************
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

#include "../utils/dtype_supported_on_device.h"
#include "util.h"

namespace {
using Outputs =
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
}

struct FusedDropoutHpuOpTest : public HpuOpTestUtil {
  auto computeOutputsUsingDefaultGen(
      std::initializer_list<long int> sizes,
      torch::ScalarType type,
      double p,
      int seed1,
      int seed2) {
    DisableRecipeCache();

    auto input = torch::randn(sizes, torch::dtype(type)).to(torch::kHPU);

    habana::getDefaultHPUGenerator().set_current_seed(seed1);
    auto dropout1 = torch::_fused_dropout(input, p);

    auto output1 = std::get<0>(dropout1).to("cpu");
    auto mask1 = std::get<1>(dropout1).to("cpu");

    habana::getDefaultHPUGenerator().set_current_seed(seed2);
    auto dropout2 = torch::_fused_dropout(input, p);

    auto output2 = std::get<0>(dropout2).to("cpu");
    auto mask2 = std::get<1>(dropout2).to("cpu");

    RestoreRecipeCache();

    return std::make_tuple(output1, output2, mask1, mask2);
  }

  auto computeOutputsUsingCustomGen(
      std::initializer_list<long int> sizes,
      torch::ScalarType type,
      double p,
      c10::optional<at::Generator> gen,
      int seed1,
      int seed2) {
    DisableRecipeCache();

    auto input = torch::randn(sizes, torch::dtype(type)).to(torch::kHPU);

    gen->set_current_seed(seed1);
    auto dropout1 = torch::_fused_dropout(input, p, gen);

    auto output1 = std::get<0>(dropout1).to("cpu");
    auto mask1 = std::get<1>(dropout1).to("cpu");

    gen->set_current_seed(seed2);
    auto dropout2 = torch::_fused_dropout(input, p, gen);

    auto output2 = std::get<0>(dropout2).to("cpu");
    auto mask2 = std::get<1>(dropout2).to("cpu");

    RestoreRecipeCache();

    return std::make_tuple(output1, output2, mask1, mask2);
  }

  void expectOutputsEquality(const Outputs& outputs) {
    auto [output1, output2, mask1, mask2] = outputs;

    EXPECT_TRUE(torch::equal(output1, output2));
    EXPECT_TRUE(torch::equal(mask1, mask2));
  }

  void expectOutputsInequality(const Outputs& outputs) {
    auto [output1, output2, mask1, mask2] = outputs;

    EXPECT_FALSE(torch::equal(output1, output2));
    EXPECT_FALSE(torch::equal(mask1, mask2));
  }
};

TEST_F(FusedDropoutHpuOpTest, DefaultGenerator_SameSeeds_ResultsEqual) {
  expectOutputsEquality(
      computeOutputsUsingDefaultGen({2, 3, 4}, torch::kFloat, 0.3, 7, 7));
}

TEST_F(FusedDropoutHpuOpTest, DefaultGenerator_DifferentSeeds_ResultsVary) {
  if (not IsDtypeSupportedOnCurrentDevice(torch::kHalf)) {
    GTEST_SKIP();
  }
  expectOutputsInequality(
      computeOutputsUsingDefaultGen({2, 3, 4, 5}, torch::kHalf, 0.7123, 6, 9));
}

TEST_F(FusedDropoutHpuOpTest, CustomGenerator_SameSeeds_ResultsEqual) {
  auto gen = at::detail::createCPUGenerator();
  expectOutputsEquality(computeOutputsUsingCustomGen(
      {7, 5, 3}, torch::kBFloat16, 0.321, gen, 9, 9));
}

TEST_F(FusedDropoutHpuOpTest, CustomGenerator_DifferentSeeds_ResultsVary) {
  auto gen = at::detail::createCPUGenerator();
  expectOutputsInequality(computeOutputsUsingCustomGen(
      {6, 4, 2, 1}, torch::kFloat, 0.91, gen, 3, 7));
}
