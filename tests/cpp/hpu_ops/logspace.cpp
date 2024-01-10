/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

TEST_F(HpuOpTest, logspace_out_1) {
  at::Scalar start = 0.0f;
  at::Scalar end = 10.0f;
  int steps = 20;
  float base = 2.0;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::logspace_outf(start, end, steps, base, expected);
  torch::logspace_outf(start, end, steps, base, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logspace_out_2) {
  at::Scalar start = 10.0f;
  at::Scalar end = 0.0f;
  int steps = 2;
  float base = 2.0;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty({2}, dtype);
  auto result = torch::empty({2}, torch::TensorOptions(dtype).device("hpu"));
  torch::logspace_outf(start, end, steps, base, expected);
  torch::logspace_outf(start, end, steps, base, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logspace_out_3) {
  at::Scalar end = -40.0f;
  at::Scalar start = -10.0f;
  int steps = 10;
  float base = 2.0;

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::logspace_outf(start, end, steps, base, expected);
  torch::logspace_outf(start, end, steps, base, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logspace_out_4) {
  at::Scalar end = -40.0;
  at::Scalar start = -10.0;
  int steps = 10;
  float base = 2.0;

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::logspace_outf(start, end, steps, base, expected);
  torch::logspace_outf(start, end, steps, base, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logspace_out_5) {
  at::Scalar start = 0.0f;
  at::Scalar end = 10.0f;
  int steps = 2;
  float base = 2.0;

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::logspace_outf(start, end, steps, base, expected);
  torch::logspace_outf(start, end, steps, base, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logspace_out_6) {
  at::Scalar start = 10.0f;
  at::Scalar end = 0.0f;
  int steps = 20;
  float base = 2.0;

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::logspace_outf(start, end, steps, base, expected);
  torch::logspace_outf(start, end, steps, base, result);
  Compare(expected, result, 2e-2, 1e-5);
}
