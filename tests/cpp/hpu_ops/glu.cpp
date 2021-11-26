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

TEST_F(HpuOpTest, glu) {
  GenerateInputs(1, {{4, 8, 16, 32}});

  auto expected = torch::glu(GetCpuInput(0), /*dim*/ -3);
  auto result = torch::glu(GetHpuInput(0), /*dim*/ -3);

  Compare(expected, result);
}

TEST_F(HpuOpTest, glu_out) {
  GenerateInputs(1, {{4, 8, 16, 32, 32}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::glu_out(expected, GetCpuInput(0), /*dim*/ 2);
  torch::glu_out(result, GetHpuInput(0), /*dim*/ 2);

  Compare(expected, result);
}

TEST_F(HpuOpTest, glu_bwd) {
  GenerateInputs(2, {{4, 8, 16, 32}, {2, 8, 16, 32}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected =
      torch::glu_backward(GetCpuInput(1), GetCpuInput(0), /*dim*/ 0);
  auto result = torch::glu_backward(GetHpuInput(1), GetHpuInput(0), /*dim*/ 0);

  Compare(expected, result);
}

TEST_F(HpuOpTest, glu_bwd_out) {
  GenerateInputs(2, {{4, 8, 16, 32, 32}, {4, 8, 16, 16, 32}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::glu_backward_out(expected, GetCpuInput(1), GetCpuInput(0), /*dim*/ 3);
  torch::glu_backward_out(result, GetHpuInput(1), GetHpuInput(0), /*dim*/ 3);

  Compare(expected, result);
}