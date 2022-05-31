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
