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

TEST_F(HpuOpTest, tanh_backward_out) {
  GenerateInputs(1);
  torch::ScalarType dtype = torch::kFloat;
  torch::Tensor grad_out = torch::ones({2, 3, 2});
  grad_out = grad_out.to(dtype);
  auto hgrad_out = grad_out.to(torch::kHPU, dtype);
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::tanh_backward_outf(grad_out, GetCpuInput(0), expected);
  torch::tanh_backward_outf(hgrad_out, GetHpuInput(0), result);
  Compare(expected, result);
}
