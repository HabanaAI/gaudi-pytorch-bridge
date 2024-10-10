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

TEST_F(HpuOpTest, sigmoid) {
  GenerateInputs(1, {{1, 2}});

  auto expected = torch::sigmoid(GetCpuInput(0));
  auto result = torch::sigmoid(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, sigmoid_) {
  GenerateInputs(1, {{1, 2}});

  GetCpuInput(0).sigmoid_();
  GetHpuInput(0).sigmoid_();

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, sigmoid_out) {
  GenerateInputs(1, {{2, 3, 5}});

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sigmoid_outf(GetCpuInput(0), expected);
  torch::sigmoid_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sigmoid_backward_out) {
  GenerateInputs(1, {{2, 3, 5}});

  torch::Tensor grad_out = torch::ones({2, 3, 5});

  torch::ScalarType dtype = torch::kFloat;
  grad_out = grad_out.to(dtype);

  auto hgrad_out = grad_out.to(torch::kHPU, dtype);
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sigmoid_backward_outf(grad_out, GetCpuInput(0), expected);
  torch::sigmoid_backward_outf(hgrad_out, GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, hardsigmoid_bwd_out) {
  GenerateInputs(2);

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::hardsigmoid_backward_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::hardsigmoid_backward_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}