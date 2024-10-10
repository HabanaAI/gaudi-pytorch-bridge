/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, any_out) {
  GenerateInputs(1, {{4, 8, 16, 32}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::any_out(expected, GetCpuInput(0));
  torch::any_out(result, GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, any_keepdim) {
  GenerateInputs(1, {{4, 8, 16, 32}});

  auto expected = torch::any(GetCpuInput(0), 0, true);
  auto result = torch::any(GetHpuInput(0), 0, true);

  Compare(expected, result);
}

TEST_F(HpuOpTest, any_bf16) {
  GenerateInputs(1, {{4, 8, 16}}, {torch::kBFloat16});

  auto expected = torch::any(GetCpuInput(0), 1, false);
  auto result = torch::any(GetHpuInput(0), 1, false);

  Compare(expected, result);
}

// input inf
TEST_F(HpuOpTest, any_inf) {
  float pos_inf = std::numeric_limits<int>::infinity();
  float neg_inf = -std::numeric_limits<int>::infinity();
  auto tensor1 = torch::tensor({pos_inf, neg_inf});
  auto tensor2 = torch::tensor({pos_inf, neg_inf}, "hpu");

  auto expected = torch::any(tensor1, 0, true);
  auto result = torch::any(tensor2, 0, true);

  Compare(expected, result);
}