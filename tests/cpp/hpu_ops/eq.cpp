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

TEST_F(HpuOpTest, eq_scalar) {
  GenerateInputs(1, {{2, 3, 4}});
  float other = 1.1;

  GetCpuInput(0).eq_(other);
  GetHpuInput(0).eq_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, eq_tensor) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}});

  GetCpuInput(0).eq_(GetCpuInput(1));
  GetHpuInput(0).eq_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, equal_f32) {
  auto ones = torch::ones({12}, "hpu");
  auto zeros = torch::zeros({12}, "hpu");
  EXPECT_FALSE(torch::equal(ones, zeros));
}

TEST_F(HpuOpTest, equal_bf16) {
  auto ones = torch::ones(
      {2, 3, 4}, torch::TensorOptions(torch::kBFloat16).device("hpu"));
  EXPECT_TRUE(torch::equal(ones, ones));
}

TEST_F(HpuOpTest, equal_i32) {
  auto ones =
      torch::ones({2, 3, 4}, torch::TensorOptions(torch::kInt32).device("hpu"));
  EXPECT_TRUE(torch::equal(ones, ones));
}

TEST_F(HpuOpTest, equal_i8) {
  auto ones =
      torch::ones({8, 4, 4}, torch::TensorOptions(torch::kInt8).device("hpu"));
  auto zeros =
      torch::zeros({8, 4, 4}, torch::TensorOptions(torch::kInt8).device("hpu"));
  EXPECT_FALSE(torch::equal(ones, zeros));
}

TEST_F(HpuOpTest, equal_u8) {
  auto ones = torch::ones(
      {8, 4, 4, 3}, torch::TensorOptions(torch::kUInt8).device("hpu"));
  EXPECT_TRUE(torch::equal(ones, ones));
}
