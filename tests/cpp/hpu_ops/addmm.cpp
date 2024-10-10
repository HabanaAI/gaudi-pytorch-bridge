/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

TEST_F(HpuOpTest, addmm_inplace_1) {
  constexpr int n = 15;
  constexpr int m = 30;
  constexpr int p = 45;
  GenerateInputs(3, {{n, p}, {n, m}, {m, p}});

  GetCpuInput(0).addmm_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 4.0031, /*alpha*/ 3.0);
  GetHpuInput(0).addmm_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 4.0031, /*alpha*/ 3.0);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

/**
 * Default tolerance will fail for BFloat16
 * Issue Raised: https://jira.habana-labs.com/browse/SW-67286
 */
TEST_F(HpuOpTest, addmm_inplace_2) {
  constexpr int n = 32;
  constexpr int m = 24;
  constexpr int p = 16;
  GenerateInputs(3, {{n, p}, {n, m}, {m, p}}, {torch::kBFloat16});
  GetCpuInput(0).addmm_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 2.0, /*alpha*/ 3.0);
  GetHpuInput(0).addmm_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 2.0, /*alpha*/ 3.0);
  Compare(GetCpuInput(0), GetHpuInput(0), 2e-2, 2e-2);
}

TEST_F(HpuOpTest, addmm_inplace_3) {
  constexpr int n = 32;
  constexpr int m = 24;
  constexpr int p = 16;
  GenerateInputs(3, {{n, p}, {n, m}, {m, p}});

  GetCpuInput(0).addmm_(
      GetCpuInput(1), GetCpuInput(2), /*beta*/ 2.0, /*alpha*/ 3.0);
  GetHpuInput(0).addmm_(
      GetHpuInput(1), GetHpuInput(2), /*beta*/ 2.0, /*alpha*/ 3.0);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, addmm_out_broadcast_1) {
  constexpr int n = 20;
  constexpr int m = 30;
  constexpr int p = 40;
  GenerateInputs(3, {{1, 1}, {n, m}, {m, p}});

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");
  torch::addmm_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 0.0,
      /*alpha*/ 4.0,
      expected);
  torch::addmm_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 0.0,
      /*alpha*/ 4.0,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addmm_out_broadcast_2) {
  constexpr int n = 20;
  constexpr int m = 30;
  constexpr int p = 40;
  GenerateInputs(3, {{1}, {n, m}, {m, p}});

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");
  torch::addmm_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      /*beta*/ 6.0,
      /*alpha*/ 8.0,
      expected);
  torch::addmm_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      /*beta*/ 6.0,
      /*alpha*/ 8.0,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addmmTest) {
  constexpr int n = 2;
  constexpr int m = 2;
  constexpr int p = 2;
  const c10::Scalar alpha = 1, beta = 1;
  GenerateInputs(3, {{n, p}, {n, m}, {m, p}});

  torch::Tensor expected =
      torch::addmm(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result =
      torch::addmm(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addmmTestAlphaBeta) {
  constexpr int n = 2;
  constexpr int m = 3;
  constexpr int p = 4;
  const c10::Scalar alpha = 1.2, beta = 1.8;
  GenerateInputs(3, {{n, p}, {n, m}, {m, p}});

  torch::Tensor expected =
      torch::addmm(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result =
      torch::addmm(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addmmTestAlpha) {
  constexpr int n = 3;
  constexpr int m = 4;
  constexpr int p = 5;
  const c10::Scalar alpha = 3.8, beta = 0.0;
  GenerateInputs(3, {{n, p}, {n, m}, {m, p}});

  torch::Tensor expected =
      torch::addmm(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result =
      torch::addmm(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addmmTestAlphaBF16) {
  constexpr int n = 23;
  constexpr int m = 34;
  constexpr int p = 15;
  const c10::Scalar alpha = 3.8, beta = 0.0;
  GenerateInputs(3, {{n, p}, {n, m}, {m, p}}, {torch::kBFloat16});

  torch::Tensor expected =
      torch::addmm(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result =
      torch::addmm(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);
  Compare(expected, result, 2e-2, 2e-2);
}

TEST_F(HpuOpTest, addmmTestBeta) {
  constexpr int n = 3;
  constexpr int m = 4;
  constexpr int p = 5;
  const c10::Scalar alpha = 1.0, beta = 3.2;
  GenerateInputs(3, {{n, p}, {n, m}, {m, p}});

  torch::Tensor expected =
      torch::addmm(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result =
      torch::addmm(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);
  Compare(expected, result);
}

TEST_F(HpuOpTest, addmmTestAlphaBetaBroadcast1) {
  constexpr int n = 3;
  constexpr int m = 5;
  constexpr int p = 7;
  const c10::Scalar alpha = 1.9, beta = 3.2;
  GenerateInputs(3, {{1, p}, {n, m}, {m, p}});

  torch::Tensor expected =
      torch::addmm(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result =
      torch::addmm(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);

  Compare(expected, result);
}

TEST_F(HpuOpTest, addmmTestAlphaBetaBroadcast2) {
  constexpr int n = 3;
  constexpr int m = 5;
  constexpr int p = 7;
  const c10::Scalar alpha = 1.9, beta = 3.2;
  GenerateInputs(3, {{n, 1}, {n, m}, {m, p}});

  torch::Tensor expected =
      torch::addmm(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), beta, alpha);

  auto result =
      torch::addmm(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), beta, alpha);

  Compare(expected, result);
}
