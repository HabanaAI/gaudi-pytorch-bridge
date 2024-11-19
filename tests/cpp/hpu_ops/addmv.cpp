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

TEST_F(HpuOpTest, addmv_usual_1) {
  int rowCount = GenerateScalar<int>(25, 35);
  int columnCount = GenerateScalar<int>(12, 24);
  GenerateInputs(3, {{rowCount}, {rowCount, columnCount}, {columnCount}});

  auto expected = torch::addmv(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 11.0);
  auto result = torch::addmv(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 11.0);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addmv_usual_2) {
  int rowCount = GenerateScalar<int>(100, 200);
  int columnCount = GenerateScalar<int>(10, 20);
  GenerateInputs(3, {{rowCount}, {rowCount, columnCount}, {columnCount}});

  auto expected = torch::addmv(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
  auto result = torch::addmv(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));
  Compare(expected, result);
}

/**
 * Default tolerance will fail for BFloat16
 * Issue Raised: https://jira.habana-labs.com/browse/SW-67283
 */
TEST_F(HpuOpTest, addmv_usual_3) {
  int rowCount = GenerateScalar<int>(5, 15);
  int columnCount = GenerateScalar<int>(10, 20);
  GenerateInputs(
      3,
      {{rowCount}, {rowCount, columnCount}, {columnCount}},
      {torch::kBFloat16});

  auto expected = torch::addmv(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 3.0,
      /*alpha*/ 1.0);
  auto result = torch::addmv(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 3.0,
      /*alpha*/ 1.0);
  Compare(expected, result, 1e-2, 1e-2);
}

TEST_F(HpuOpTest, addmv_usual_4) {
  int rowCount = GenerateScalar<int>(5, 15);
  int columnCount = GenerateScalar<int>(10, 20);
  GenerateInputs(
      3,
      {{rowCount}, {rowCount, columnCount}, {columnCount}},
      {torch::kBFloat16});

  auto& inputCPU = GetCpuInput(0);
  inputCPU[0] = NAN;

  auto& inputHPU = GetHpuInput(0);
  inputHPU[0] = NAN;

  auto expected = torch::addmv(
      inputCPU,
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 0.0,
      /*alpha*/ 1.0);
  auto result = torch::addmv(
      inputHPU,
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 0.0,
      /*alpha*/ 1.0);
  Compare(expected, result, 1e-2, 1e-2);
}

TEST_F(HpuOpTest, addmv_usual_broadcast_1) {
  int rowCount = GenerateScalar<int>(50, 150);
  int columnCount = GenerateScalar<int>(20, 45);
  GenerateInputs(3, {{1}, {rowCount, columnCount}, {columnCount}});

  auto expected = torch::addmv(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ -5.0,
      /*alpha*/ 12.0);
  auto result = torch::addmv(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ -5.0,
      /*alpha*/ 12.0);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addmv_usual_broadcast_2) {
  int rowCount = GenerateScalar<int>(50, 150);
  int columnCount = GenerateScalar<int>(20, 45);
  GenerateInputs(3, {{1}, {rowCount, columnCount}, {columnCount}});

  auto expected = torch::addmv(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ -0.5,
      /*alpha*/ 0.12);
  auto result = torch::addmv(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ -0.5,
      /*alpha*/ 0.12);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addmv_inplace_1) {
  int rowCount = GenerateScalar<int>(25, 75);
  int columnCount = GenerateScalar<int>(60, 120);
  GenerateInputs(3, {{rowCount}, {rowCount, columnCount}, {columnCount}});

  GetCpuInput(0).addmv_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 4.0031, /*alpha*/ 3.0);
  GetHpuInput(0).addmv_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 4.0031, /*alpha*/ 3.0);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, addmv_inplace_2) {
  int rowCount = GenerateScalar<int>(28, 56);
  int columnCount = GenerateScalar<int>(32, 64);
  GenerateInputs(3, {{rowCount}, {rowCount, columnCount}, {columnCount}});

  GetCpuInput(0).addmv_(GetCpuInput(1), GetCpuInput(2));
  GetHpuInput(0).addmv_(GetHpuInput(1), GetHpuInput(2));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, addmv_out_broadcast) {
  int rowCount = GenerateScalar<int>(46, 51);
  int columnCount = GenerateScalar<int>(64, 99);
  GenerateInputs(3, {{1}, {rowCount, columnCount}, {columnCount}});

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");
  torch::addmv_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 4.0,
      /*alpha*/ -8.0,
      expected);
  torch::addmv_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 4.0,
      /*alpha*/ -8.0,
      result);
  Compare(expected, result);
}
