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
