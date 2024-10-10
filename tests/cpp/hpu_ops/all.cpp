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

TEST_F(HpuOpTest, all_dim_out) {
  GenerateInputs(1, {{4, 8, 16, 32}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::all_out(expected, GetCpuInput(0), /*dim*/ 1, /*keepdim*/ true);
  torch::all_out(result, GetHpuInput(0), /*dim*/ 1, /*keepdim*/ true);

  Compare(expected, result);
}

TEST_F(HpuOpTest, all_keepdim_false_out) {
  GenerateInputs(1, {{12, 5, 6, 3}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::all_out(expected, GetCpuInput(0), /*dim*/ 0, /*keepdim*/ false);
  torch::all_out(result, GetHpuInput(0), /*dim*/ 0, /*keepdim*/ false);

  Compare(expected, result);
}

TEST_F(HpuOpTest, all_neg_dim_out) {
  GenerateInputs(1, {{3, 11, 2, 4}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::all_out(expected, GetCpuInput(0), /*dim*/ -1, /*keepdim*/ true);
  torch::all_out(result, GetHpuInput(0), /*dim*/ -1, /*keepdim*/ true);

  Compare(expected, result);
}

TEST_F(HpuOpTest, all_out) {
  GenerateInputs(1, {{4, 8, 16}}, {torch::kBool});
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::all_out(expected, GetCpuInput(0));
  torch::all_out(result, GetHpuInput(0));

  Compare(expected, result);
}
