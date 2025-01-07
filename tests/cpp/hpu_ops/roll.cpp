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

TEST_F(HpuOpTest, roll) {
  constexpr unsigned int dim0 = 4, dim1 = 3;
  GenerateInputs(1, {{dim0, dim1}});

  std::srand((unsigned int)-1);
  std::array<int64_t, 2> shift = {-1 * std::rand(), std::rand()};
  std::array<int64_t, 2> axis = {0, 1};

  auto expected = torch::roll(GetCpuInput(0), shift, axis);
  auto result = torch::roll(GetHpuInput(0), shift, axis);

  Compare(expected, result);
}

TEST_F(HpuOpTest, roll_1d) {
  constexpr unsigned int dim0 = 4;
  GenerateInputs(1, {{dim0}});

  std::srand((unsigned int)-1);
  std::array<int64_t, 1> shift = {-1 * std::rand()};
  std::array<int64_t, 1> axis = {0};

  auto expected = torch::roll(GetCpuInput(0), shift, axis);
  auto result = torch::roll(GetHpuInput(0), shift, axis);

  Compare(expected, result);
}

TEST_F(HpuOpTest, roll_5d) {
  constexpr unsigned int dim0 = 4, dim1 = 3, dim2 = 1, dim3 = 2, dim4 = 6;
  GenerateInputs(1, {{dim0, dim1, dim2, dim3, dim4}});

  std::srand((unsigned int)-1);
  std::array<int64_t, 5> shift = {
      -1 * std::rand(),
      std::rand(),
      std::rand(),
      -1 * std::rand(),
      std::rand()};
  std::array<int64_t, 5> axis = {2, 1, 0, 4, 3};

  auto expected = torch::roll(GetCpuInput(0), shift, axis);
  auto result = torch::roll(GetHpuInput(0), shift, axis);

  Compare(expected, result);
}

TEST_F(HpuOpTest, roll_bf16) {
  constexpr unsigned int dim0 = 4, dim1 = 3;
  GenerateInputs(1, {{dim0, dim1}}, {torch::kBFloat16});

  std::srand((unsigned int)-1);
  std::array<int64_t, 2> shift = {-1 * std::rand(), std::rand()};
  std::array<int64_t, 2> axis = {0, 1};

  auto expected = torch::roll(GetCpuInput(0), shift, axis);
  auto result = torch::roll(GetHpuInput(0), shift, axis);

  Compare(expected, result);
}

TEST_F(HpuOpTest, roll_u8) {
  constexpr unsigned int dim0 = 4, dim1 = 5;
  GenerateInputs(1, {{dim0, dim1}}, {torch::kUInt8});

  std::srand((unsigned int)-1);
  std::array<int64_t, 2> shift = {-1 * std::rand(), std::rand()};
  std::array<int64_t, 2> axis = {0, 1};

  auto expected = torch::roll(GetCpuInput(0), shift, axis);
  auto result = torch::roll(GetHpuInput(0), shift, axis);

  Compare(expected, result);
}

TEST_F(HpuOpTest, roll_1d_axis_none) {
  constexpr unsigned int dim0 = 4;
  GenerateInputs(1, {{dim0}});

  std::srand((unsigned int)-1);
  std::array<int64_t, 1> shift = {-1 * std::rand()};
  std::array<int64_t, 0> axis = {};

  auto expected = torch::roll(GetCpuInput(0), shift, axis);
  auto result = torch::roll(GetHpuInput(0), shift, axis);

  Compare(expected, result);
}

TEST_F(HpuOpTest, roll_5d_axis_none) {
  constexpr unsigned int dim0 = 4, dim1 = 3, dim2 = 1, dim3 = 2, dim4 = 6;
  GenerateInputs(1, {{dim0, dim1, dim2, dim3, dim4}});

  std::srand((unsigned int)-1);
  std::array<int64_t, 1> shift = {std::rand()};
  std::array<int64_t, 0> axis = {};

  auto expected = torch::roll(GetCpuInput(0), shift, axis);
  auto result = torch::roll(GetHpuInput(0), shift, axis);

  Compare(expected, result);
}