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

TEST_F(HpuOpTest, softmax_out_float) {
  GenerateInputs(1);

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::_softmax_outf(
      GetCpuInput(0), /*dim*/ {}, /*half_to_float*/ false, expected);
  torch::_softmax_outf(
      GetHpuInput(0), /*dim*/ {}, /*half_to_float*/ false, result);

  Compare(expected, result);
}

/**
 * BFloat16 testcases added below will fail for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-68069
 **/
TEST_F(HpuOpTest, softmax_out_bfloat) {
  GenerateInputs(1, torch::kBFloat16);

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::_softmax_outf(
      GetCpuInput(0), /*dim*/ 2, /*half_to_float*/ false, expected);
  torch::_softmax_outf(
      GetHpuInput(0), /*dim*/ 2, /*half_to_float*/ false, result);

  Compare(expected, result, 1e-03, 4e-03);
}

/**
 * BFloat16 testcases added below will fail for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-68069
 **/
TEST_F(HpuOpTest, softmax_out_bfloat_negdim) {
  GenerateInputs(1, torch::kBFloat16);

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::_softmax_outf(
      GetCpuInput(0), /*dim*/ -2, /*half_to_float*/ false, expected);
  torch::_softmax_outf(
      GetHpuInput(0), /*dim*/ -2, /*half_to_float*/ false, result);

  Compare(expected, result, 1e-03, 4e-03);
}

TEST_F(HpuOpTest, softmax_bwd_out_float) {
  const std::vector<int64_t> size = {5, 3, 8};
  GenerateInputs(3, {size, size, size});

  auto hgrad_out = GetCpuInput(1).to(torch::kHPU);
  auto houtput = GetCpuInput(2).to(torch::kHPU);
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::_softmax_backward_data_outf(
      GetCpuInput(1), GetCpuInput(2), /*dim*/ -1, GetCpuInput(0).scalar_type(), expected);
  torch::_softmax_backward_data_outf(
      hgrad_out, houtput, /*dim*/ -1, GetHpuInput(0).scalar_type(), result);

  Compare(expected, result);
}

/**
 * BFloat16 testcases added below will fail for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-68069
 */
TEST_F(HpuOpTest, softmax_bwd_out_bfloat) {
  torch::ScalarType dtype = torch::kBFloat16;
  const std::vector<int64_t> size = {2, 1, 4, 3};

  GenerateInputs(3, {size, size, size}, dtype);

  auto hgrad_out = GetCpuInput(1).to(torch::kHPU, dtype);
  auto houtput = GetCpuInput(2).to(torch::kHPU, dtype);

  auto expected = torch::empty({0}, dtype);
  auto result = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

  torch::_softmax_backward_data_outf(
      GetCpuInput(1), GetCpuInput(2), /*dim*/ 3, GetCpuInput(0).scalar_type(), expected);
  torch::_softmax_backward_data_outf(
      hgrad_out, houtput, /*dim*/ 3, GetHpuInput(0).scalar_type(), result);

  Compare(expected, result, 1e-03, 6e-02);
}