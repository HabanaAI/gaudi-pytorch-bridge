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

TEST_F(HpuOpTest, DISABLED_addbmm_inplace_1) {
  // TODO: SW-172900
  constexpr int n = 15;
  constexpr int m = 30;
  constexpr int p = 45;
  constexpr int batch = 10;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}});

  GetCpuInput(0).addbmm_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 4.0031, /*alpha*/ 3.0);
  GetHpuInput(0).addbmm_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 4.0031, /*alpha*/ 3.0);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, addbmm_inplace_2) {
  constexpr int n = 15;
  constexpr int m = 30;
  constexpr int p = 45;
  constexpr int batch = 10;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}});

  GetCpuInput(0).addbmm_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 0.0, /*alpha*/ 0.0);
  GetHpuInput(0).addbmm_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 0.0, /*alpha*/ 0.0);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

/**
 * Default tolerance will fail for BFloat16
 * Issue Raised: https://jira.habana-labs.com/browse/SW-67286
 */
TEST_F(HpuOpTest, addbmm_inplace_3) {
  constexpr int n = 32;
  constexpr int m = 24;
  constexpr int p = 16;
  constexpr int batch = 10;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}}, {torch::kBFloat16});
  GetCpuInput(0).addbmm_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 2.0, /*alpha*/ 3.0);
  GetHpuInput(0).addbmm_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 2.0, /*alpha*/ 3.0);
  Compare(GetCpuInput(0), GetHpuInput(0), 2e-1, 1e-1);
}

TEST_F(HpuOpTest, addbmm_inplace_4) {
  constexpr int n = 32;
  constexpr int m = 24;
  constexpr int p = 16;
  constexpr int batch = 10;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}});

  GetCpuInput(0).addbmm_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 2.0, /*alpha*/ 3.0);
  GetHpuInput(0).addbmm_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 2.0, /*alpha*/ 3.0);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, addbmm_out_broadcast_1) {
  constexpr int n = 20;
  constexpr int m = 30;
  constexpr int p = 40;
  constexpr int batch = 10;
  GenerateInputs(3, {{1, 1}, {batch, n, m}, {batch, m, p}});

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");
  torch::addbmm_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 0.0,
      /*alpha*/ 4.0,
      expected);
  torch::addbmm_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 0.0,
      /*alpha*/ 4.0,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addbmm_out_broadcast_2) {
  constexpr int n = 20;
  constexpr int m = 30;
  constexpr int p = 40;
  constexpr int batch = 10;
  GenerateInputs(3, {{1}, {batch, n, m}, {batch, m, p}});

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");
  torch::addbmm_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 6.0,
      /*alpha*/ 8.0,
      expected);
  torch::addbmm_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 6.0,
      /*alpha*/ 8.0,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addbmmTest) {
  constexpr int n = 2;
  constexpr int m = 2;
  constexpr int p = 2;
  constexpr int batch = 10;
  const c10::Scalar alpha = 0, beta = 1;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}});

  torch::Tensor expected = torch::addbmm(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result = torch::addbmm(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addbmmTestAlphaBeta) {
  constexpr int n = 2;
  constexpr int m = 3;
  constexpr int p = 4;
  constexpr int batch = 10;
  const c10::Scalar alpha = 1.2, beta = 1.8;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}});

  torch::Tensor expected = torch::addbmm(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result = torch::addbmm(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addbmmTestAlpha) {
  constexpr int n = 3;
  constexpr int m = 4;
  constexpr int p = 5;
  constexpr int batch = 10;
  const c10::Scalar alpha = 3.8, beta = 0.0;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}});

  torch::Tensor expected = torch::addbmm(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result = torch::addbmm(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addbmmTestAlphaBF16) {
  constexpr int n = 23;
  constexpr int m = 34;
  constexpr int p = 15;
  constexpr int batch = 10;
  const c10::Scalar alpha = 3.8, beta = 0.0;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}}, {torch::kBFloat16});

  torch::Tensor expected = torch::addbmm(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result = torch::addbmm(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);
  Compare(expected, result, 2e-1, 1e-1);
}

TEST_F(HpuOpTest, addbmmTestBeta) {
  constexpr int n = 3;
  constexpr int m = 4;
  constexpr int p = 5;
  constexpr int batch = 10;
  const c10::Scalar alpha = 1.0, beta = 3.2;
  GenerateInputs(3, {{n, p}, {batch, n, m}, {batch, m, p}});

  torch::Tensor expected = torch::addbmm(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result = torch::addbmm(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addbmmTestAlphaBetaBroadcast1) {
  constexpr int n = 3;
  constexpr int m = 5;
  constexpr int p = 7;
  constexpr int batch = 10;
  const c10::Scalar alpha = 1.9, beta = 3.2;
  GenerateInputs(3, {{1, p}, {batch, n, m}, {batch, m, p}});

  torch::Tensor expected = torch::addbmm(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result = torch::addbmm(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addbmmTestAlphaBetaBroadcast2) {
  constexpr int n = 3;
  constexpr int m = 5;
  constexpr int p = 7;
  constexpr int batch = 10;
  const c10::Scalar alpha = 1.9, beta = 3.2;
  GenerateInputs(3, {{n, 1}, {batch, n, m}, {batch, m, p}});

  torch::Tensor expected = torch::addbmm(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result = torch::addbmm(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);

  Compare(expected, result);
}
