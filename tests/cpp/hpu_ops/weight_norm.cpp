/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

TEST_F(HpuOpTest, WeightNormTest) {
  GenerateInputs(2, {{16, 32, 64}, {1, 1, 64}}, {at::kFloat, at::kFloat});
  int64_t dim = 2;

  // CPU Run
  at::Tensor expected = at::_weight_norm(GetCpuInput(0), GetCpuInput(1), dim);
  // HPU Run
  at::Tensor result = at::_weight_norm(GetHpuInput(0), GetHpuInput(1), dim);
  // Compare CPU vs HPU
  Compare(expected, result);
}

TEST_F(HpuOpTest, WeightNormDim0Test) {
  GenerateInputs(2, {{128, 64}, {128, 1}}, {at::kFloat, at::kFloat});
  int64_t dim = 0;

  // CPU Run
  at::Tensor expected = at::_weight_norm(GetCpuInput(0), GetCpuInput(1), dim);
  // HPU Run
  at::Tensor result = at::_weight_norm(GetHpuInput(0), GetHpuInput(1), dim);
  // Compare CPU vs HPU
  Compare(expected, result);
}

TEST_F(HpuOpTest, WeightNormBackwardExecute) {
  GenerateInputs(
      4,
      {{32, 16}, {32, 16}, {32, 1}, {32, 1}},
      {
          at::kFloat,
          at::kFloat,
          at::kFloat,
          at::kFloat,
      });
  int64_t dim = 0;
  // CPU Run
  auto expected = at::_weight_norm_interface_backward(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), GetCpuInput(3), dim);
  // HPU Run
  auto result = at::_weight_norm_interface_backward(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), GetHpuInput(3), dim);
  // Compare CPU vs HPU
  Compare(expected, result);
}

TEST_F(HpuOpTest, WeightNormBackwardExecute3D) {
  GenerateInputs(
      4,
      {{32, 4, 5}, {32, 4, 5}, {1, 1, 5}, {1, 1, 5}},
      {
          at::kFloat,
          at::kFloat,
          at::kFloat,
          at::kFloat,
      });
  int64_t dim = 2;
  // CPU Run
  auto expected = at::_weight_norm_interface_backward(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), GetCpuInput(3), dim);
  // HPU Run
  auto result = at::_weight_norm_interface_backward(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), GetHpuInput(3), dim);
  // Compare CPU vs HPU
  Compare(expected, result);
}
