/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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