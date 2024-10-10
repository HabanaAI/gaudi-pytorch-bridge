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

TEST_F(HpuOpTest, LeakyRelu_bwd_out) {
  GenerateInputs(2);
  float neg_slope = GenerateScalar<float>(1e-2, 1e2);
  bool self_is_result = false;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::leaky_relu_backward_outf(
      GetCpuInput(0), GetCpuInput(1), neg_slope, self_is_result, expected);
  torch::leaky_relu_backward_outf(
      GetHpuInput(0), GetHpuInput(1), neg_slope, self_is_result, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, LeakyRelu_bwd_out_bf16) {
  GenerateInputs(2, {torch::kBFloat16});
  float neg_slope = GenerateScalar<float>(1e-2, 1e2);
  bool self_is_result = false;

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::leaky_relu_backward_outf(
      GetCpuInput(0), GetCpuInput(1), neg_slope, self_is_result, expected);
  torch::leaky_relu_backward_outf(
      GetHpuInput(0), GetHpuInput(1), neg_slope, self_is_result, result);
  Compare(expected, result, 0.01, 0.01);
}