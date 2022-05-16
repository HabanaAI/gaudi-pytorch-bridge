/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, _foreach_abs_) {
  int n = 20;
  GenerateInputs(n);

  std::vector<at::Tensor> cpu_in, hpu_in;
  for (int i = 0; i < n; ++i) {
    cpu_in.push_back(GetCpuInput(i));
    hpu_in.push_back(GetHpuInput(i));
  }

  at::_foreach_abs_(cpu_in);
  at::_foreach_abs_(hpu_in);

  for (int i = 0; i < n; ++i) {
    Compare(cpu_in[i], hpu_in[i], 0, 0);
  }
}

TEST_F(HpuOpTest, _foreach_abs) {
  int n = 20;
  GenerateInputs(n);

  std::vector<at::Tensor> cpu_in, hpu_in;
  for (int i = 0; i < n; ++i) {
    cpu_in.push_back(GetCpuInput(i));
    hpu_in.push_back(GetHpuInput(i));
  }

  auto exp = at::_foreach_abs(cpu_in);
  auto res = at::_foreach_abs(hpu_in);

  for (int i = 0; i < n; ++i) {
    Compare(exp[i], res[i], 0, 0);
  }
}
