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

TEST_F(HpuOpTest, log_sigmoid_fwd) {
  GenerateInputs(1);

  auto exp = torch::log_sigmoid_forward(GetCpuInput(0));
  auto res = torch::log_sigmoid_forward(GetHpuInput(0));
  Compare(std::get<0>(exp), std::get<0>(res));
  Compare(std::get<1>(exp), std::get<1>(res));
}

TEST_F(HpuOpTest, log_sigmoid_fwd_out) {
  GenerateInputs(1);

  auto out = torch::empty(0);
  auto hout = torch::empty(0, c10::kHPU);
  auto buffer = torch::empty(0);
  auto hbuffer = torch::empty(0, c10::kHPU);

  torch::log_sigmoid_forward_outf(GetCpuInput(0), out, buffer);
  torch::log_sigmoid_forward_outf(GetHpuInput(0), hout, hbuffer);
  Compare(out, hout);
  Compare(buffer, hbuffer);
}

TEST_F(HpuOpTest, log_sigmoid_bwd) {
  GenerateInputs(3);

  auto expected = torch::log_sigmoid_backward(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
  auto result = torch::log_sigmoid_backward(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));
  Compare(expected, result);
}

TEST_F(HpuOpTest, log_sigmoid_bwd_out) {
  GenerateInputs(3);

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::log_sigmoid_backward_outf(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), expected);
  torch::log_sigmoid_backward_outf(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), result);

  Compare(expected, result);
}