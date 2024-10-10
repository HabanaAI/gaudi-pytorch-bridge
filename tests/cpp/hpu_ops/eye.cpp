/******************************************************************************
 * Copyright (C) 2021-2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, eye) {
  int n = GenerateScalar<int>(1, 8);
  auto expected = torch::eye(n);
  auto result = torch::eye(n, "hpu");
  Compare(expected, result);
}

TEST_F(HpuOpTest, eye_m) {
  int n = GenerateScalar<int>(1, 8);
  int m = GenerateScalar<int>(1, 8);
  auto expected = torch::eye(n, m);
  auto result = torch::eye(n, m, "hpu");
  Compare(expected, result);
}

TEST_F(HpuOpTest, eye_out_F32) {
  int64_t n = 3;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({n, n}, dtype);
  auto result = torch::empty({n, n}, torch::TensorOptions(dtype).device("hpu"));

  torch::eye_outf(n, expected);
  torch::eye_outf(n, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, eye_out_I32) {
  int64_t n = 3;
  torch::ScalarType dtype = torch::kInt32;

  auto expected = torch::empty({n, n}, dtype);
  auto result = torch::empty({n, n}, torch::TensorOptions(dtype).device("hpu"));

  torch::eye_outf(n, expected);
  torch::eye_outf(n, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, eye_m_out_F32) {
  int64_t n = 3;
  int64_t m = 2;
  torch::ScalarType dtype = torch::kInt;

  auto expected = torch::empty({n, m}, dtype);
  auto result = torch::empty({n, m}, torch::TensorOptions(dtype).device("hpu"));

  torch::eye_outf(n, m, expected);
  torch::eye_outf(n, m, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, eye_m_out_I32) {
  int64_t n = 3;
  int64_t m = 2;
  torch::ScalarType dtype = torch::kInt;

  auto expected = torch::empty({n, m}, dtype);
  auto result = torch::empty({n, m}, torch::TensorOptions(dtype).device("hpu"));

  torch::eye_outf(n, m, expected);
  torch::eye_outf(n, m, result);

  Compare(expected, result);
}