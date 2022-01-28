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

TEST_F(HpuOpTest, DISABLED_logspace_out_4) {
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

TEST_F(HpuOpTest, DISABLED_logspace_out_5) {
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
/**
 * Default tolerance will fail for BFloat16
 * Issue Raised: https://jira.habana-labs.com/browse/SW-70506
 */
TEST_F(HpuOpTest, DISABLED_logspace_out_6) {
  at::Scalar start = 10.0f;
  at::Scalar end = 0.0f;
  int steps = 20;
  float base = 2.0;

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::logspace_outf(start, end, steps, base, expected);
  torch::logspace_outf(start, end, steps, base, result);

  Compare(expected, result);
}
