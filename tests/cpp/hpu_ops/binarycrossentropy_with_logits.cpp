/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

class BCEWithLogitsParameterizedTestFixture
    : public ::testing::TestWithParam<
          std::tuple<bool, bool, at::Reduction::Reduction>>,
      public HpuOpTestUtilBase {};

TEST_P(BCEWithLogitsParameterizedTestFixture, tests) {
  bool testWeight = std::get<0>(GetParam());
  bool testPosWeight = std::get<1>(GetParam());
  at::Reduction::Reduction reductionType = std::get<2>(GetParam());

  auto input = torch::randn({5, 2, 4, 3});
  auto target = torch::randn({5, 2, 4, 3});
  auto grad_output = torch::randn({1});
  c10::optional<at::Tensor> weight =
      testWeight ? torch::randn({3}) : c10::optional<at::Tensor>();
  c10::optional<at::Tensor> pos_weight =
      testPosWeight ? torch::rand({3}) : c10::optional<at::Tensor>();

  torch::Tensor hinput = input.to(torch::kHPU);
  torch::Tensor htarget = target.to(torch::kHPU);
  torch::Tensor hgrad_out = grad_output.to(torch::kHPU);
  c10::optional<at::Tensor> hweight =
      testWeight ? weight.value().to(torch::kHPU) : c10::optional<at::Tensor>();
  c10::optional<at::Tensor> hpos_weight = testPosWeight
      ? pos_weight.value().to(torch::kHPU)
      : c10::optional<at::Tensor>();

  auto result_fwd = torch::binary_cross_entropy_with_logits(
      hinput, htarget, hweight, hpos_weight, reductionType);
  auto expected_fwd = torch::binary_cross_entropy_with_logits(
      input, target, weight, pos_weight, reductionType);

  Compare(expected_fwd, result_fwd);
}

INSTANTIATE_TEST_CASE_P(
    BCEWithLogitsTest,
    BCEWithLogitsParameterizedTestFixture,
    ::testing::Combine(
        testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(
            at::Reduction::Sum,
            at::Reduction::Mean,
            at::Reduction::None)));

TEST_F(HpuOpTest, BCELogitsFwdLossTest) {
  GenerateInputs(4, {{5, 2, 4, 3}, {5, 2, 4, 3}, {3}, {3}});

  auto expected = torch::binary_cross_entropy_with_logits(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      GetCpuInput(3),
      at::Reduction::Mean);
  auto result = torch::binary_cross_entropy_with_logits(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      GetHpuInput(3),
      at::Reduction::Mean);

  Compare(expected, result);
}
