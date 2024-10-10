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

TEST_F(HpuOpTest, hardswish) {
  GenerateInputs(1, {{2, 2}}, {torch::kFloat});

  auto hresult = torch::hardswish(GetHpuInput(0));
  auto cout = torch::hardswish(GetCpuInput(0));

  Compare(cout, hresult);
}

TEST_F(HpuOpTest, hardswish_) {
  GenerateInputs(1, {{2, 2}}, {torch::kFloat});

  auto hresult = torch::hardswish_(GetHpuInput(0));
  auto cout = torch::hardswish_(GetCpuInput(0));

  Compare(cout, hresult);
}

TEST_F(HpuOpTest, hardswish_out) {
  GenerateInputs(1, {{2, 2}}, {torch::kFloat});

  auto hresult =
      torch::empty(0, torch::TensorOptions(torch::kFloat).device("hpu"));
  auto cout = torch::empty(0, torch::kFloat);

  hresult = torch::hardswish_outf(GetHpuInput(0), hresult);
  cout = torch::hardswish_outf(GetCpuInput(0), cout);

  Compare(cout, hresult);
}

TEST_F(HpuOpTest, hardswish_backward) {
  GenerateInputs(2, {{2, 2}}, {torch::kFloat});

  auto hresult = torch::hardswish_backward(GetHpuInput(0), GetHpuInput(1));
  auto cout = torch::hardswish_backward(GetCpuInput(0), GetCpuInput(1));

  Compare(cout, hresult);
}
