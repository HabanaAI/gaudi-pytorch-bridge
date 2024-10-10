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

/**
 * elu_backward kernel requires support for scale, input_scale and is_result
 * parameters on TPC. So, All testcases below uses default value only for
 * mentioned parameters.
 * Issue raised : https://jira.habana-labs.com/browse/SW-68111
 **/
TEST_F(HpuOpTest, elu_backward_out_1) {
  GenerateInputs(2);

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::elu_backward_outf(
      GetCpuInput(0),
      /*alpha*/ 1.0,
      /*scale*/ 1.0,
      /*input_scale*/ 1.0,
      /*is_result*/ false,
      GetCpuInput(1),
      expected);
  torch::elu_backward_outf(
      GetHpuInput(0),
      /*alpha*/ 1.0,
      /*scale*/ 1.0,
      /*input_scale*/ 1.0,
      /*is_result*/ false,
      GetHpuInput(1),
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, elu_backward_out_2) {
  GenerateInputs(2);

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::elu_backward_outf(
      GetCpuInput(0),
      /*alpha*/ -0.5,
      /*scale*/ 1.0,
      /*input_scale*/ 1.0,
      /*is_result*/ false,
      GetCpuInput(1),
      expected);
  torch::elu_backward_outf(
      GetHpuInput(0),
      /*alpha*/ -0.5,
      /*scale*/ 1.0,
      /*input_scale*/ 1.0,
      /*is_result*/ false,
      GetHpuInput(1),
      result);

  Compare(expected, result);
}
