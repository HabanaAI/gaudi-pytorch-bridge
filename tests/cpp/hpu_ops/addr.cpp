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

TEST_F(HpuOpTest, addr_usual) {
  constexpr int rowCount = 20;
  constexpr int columnCount = 30;
  GenerateInputs(
      3,
      {{rowCount, columnCount}, {rowCount}, {columnCount}},
      {torch::kBFloat16});

  auto expected = torch::addr(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 0.0,
      /*alpha*/ 2.0);
  auto result = torch::addr(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 0.0,
      /*alpha*/ 2.0);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addr_usual_broadcast) {
  constexpr int rowCount = 10;
  constexpr int columnCount = 15;
  GenerateInputs(3, {{rowCount, 1}, {rowCount}, {columnCount}});

  auto expected = torch::addr(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 3.0,
      /*alpha*/ 2.0);
  auto result = torch::addr(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 3.0,
      /*alpha*/ 2.0);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addr_inplace_1) {
  constexpr int rowCount = 25;
  constexpr int columnCount = 15;
  GenerateInputs(3, {{rowCount, columnCount}, {rowCount}, {columnCount}});

  GetCpuInput(0).addr_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 4.0031, /*alpha*/ 3.0);
  GetHpuInput(0).addr_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 4.0031, /*alpha*/ 3.0);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, addr_inplace_2) {
  constexpr int rowCount = 25;
  constexpr int columnCount = 15;
  GenerateInputs(3, {{rowCount, columnCount}, {rowCount}, {columnCount}});

  GetCpuInput(0).addr_(GetCpuInput(1), GetCpuInput(2));
  GetHpuInput(0).addr_(GetHpuInput(1), GetHpuInput(2));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, addr_out_broadcast) {
  constexpr int rowCount = 20;
  constexpr int columnCount = 30;
  GenerateInputs(3, {{1, 1}, {rowCount}, {columnCount}});

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");
  torch::addr_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 2.0,
      /*alpha*/ 3.0,
      expected);
  torch::addr_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 2.0,
      /*alpha*/ 3.0,
      result);
  Compare(expected, result);
}
