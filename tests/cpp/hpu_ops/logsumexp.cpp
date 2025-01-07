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

TEST_F(HpuOpTest, logsumexp_4d_3d_keepdim) {
  GenerateInputs(1, {{2, 3, 4, 5}}, {torch::kBFloat16});
  const std::vector<int64_t> dim{};

  auto expected = torch::logsumexp(GetCpuInput(0), dim, /*keepdim*/ true);
  auto result = torch::logsumexp(GetHpuInput(0), dim, /*keepdim*/ true);

  Compare(expected, result, 1e-03, 1e-02);
}

TEST_F(HpuOpTest, logsumexp_5d_4d_keepdim) {
  GenerateInputs(1, {{6, 7, 8, 9, 10}});
  const std::vector<int64_t> dim{0, 1, 3, 2};

  auto expected = torch::logsumexp(GetCpuInput(0), dim, /*keepdim*/ true);
  auto result = torch::logsumexp(GetHpuInput(0), dim, /*keepdim*/ true);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logsumexp_2d_1d) {
  GenerateInputs(1, {{4, 5}}, {torch::kBFloat16});
  const std::vector<int64_t> dim{0};

  auto expected = torch::logsumexp(GetCpuInput(0), dim, /*keepdim*/ false);
  auto result = torch::logsumexp(GetHpuInput(0), dim, /*keepdim*/ false);

  Compare(expected, result, 1e-03, 1e-02);
}

TEST_F(HpuOpTest, logsumexp_3d_2d) {
  GenerateInputs(1, {{12, 45, 61}});
  const std::vector<int64_t> dim{-3, -2};

  auto expected = torch::logsumexp(GetCpuInput(0), dim, /*keepdim*/ false);
  auto result = torch::logsumexp(GetHpuInput(0), dim, /*keepdim*/ false);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logsumexp_5d_2d_keepdim_out) {
  GenerateInputs(1, {{5, 10, 9, 6, 7}});
  const std::vector<int64_t> dim{};
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::logsumexp_outf(GetCpuInput(0), dim, /*keepdim*/ true, expected);
  torch::logsumexp_outf(GetHpuInput(0), dim, /*keepdim*/ true, result);

  Compare(expected, result);
}

// 0d input keepdim: false
TEST_F(HpuOpTest, logsumexp_0d) {
  auto tensor1 = torch::tensor(45.0);
  auto tensor2 = torch::tensor(45.0, "hpu");
  const std::vector<int64_t> dim{};

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::logsumexp_outf(tensor1, dim, /*keepdim*/ false, expected);
  torch::logsumexp_outf(tensor2, dim, /*keepdim*/ false, result);

  Compare(expected, result);
}

// 0d input keepdim: true
TEST_F(HpuOpTest, logsumexp_0d_keepdim) {
  auto tensor1 = torch::tensor(50.0);
  auto tensor2 = torch::tensor(50.0, "hpu");
  const std::vector<int64_t> dim{0};

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::logsumexp_outf(tensor1, dim, /*keepdim*/ true, expected);
  torch::logsumexp_outf(tensor2, dim, /*keepdim*/ true, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logsumexp_4d_3d_out) {
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, {{2, 3, 4, 5}}, {dtype});
  const std::vector<int64_t> dim{0, 2, 3};

  auto expected = torch::empty((3), dtype);
  auto result = torch::empty((3), torch::TensorOptions(dtype).device("hpu"));

  torch::logsumexp_outf(GetCpuInput(0), dim, /*keepdim*/ false, expected);
  torch::logsumexp_outf(GetHpuInput(0), dim, /*keepdim*/ false, result);

  Compare(expected, result, 1e-03, 1e-02);
}
