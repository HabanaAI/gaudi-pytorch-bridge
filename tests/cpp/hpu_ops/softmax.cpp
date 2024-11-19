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

TEST_F(HpuOpTest, softmax_float) {
  GenerateInputs(1);

  auto expected =
      torch::_softmax(GetCpuInput(0), {} /*dim*/, false /*half_to_float*/);
  auto result =
      torch::_softmax(GetHpuInput(0), {} /*dim*/, false /*half_to_float*/);
  Compare(expected, result);
}

TEST_F(HpuOpTest, softmax_bf16) {
  GenerateInputs(1, torch::kBFloat16);
  auto expected =
      torch::_softmax(GetCpuInput(0), 2 /*dim*/, false /*half_to_float*/);
  auto result =
      torch::_softmax(GetHpuInput(0), 2 /*dim*/, false /*half_to_float*/);
  Compare(expected, result, 1e-03, 4e-03);
}

TEST_F(HpuOpTest, softmax_bf16_negdim) {
  GenerateInputs(1, torch::kBFloat16);
  auto expected =
      torch::_softmax(GetCpuInput(0), -2 /*dim*/, false /*half_to_float*/);
  auto result =
      torch::_softmax(GetHpuInput(0), -2 /*dim*/, false /*half_to_float*/);
  Compare(expected, result, 1e-03, 4e-03);
}

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

TEST_F(HpuOpTest, softmax_bwd_f32) {
  GenerateInputs(2);
  auto expected = torch::_softmax_backward_data(
      GetCpuInput(0), GetCpuInput(1), {} /*dim*/, GetCpuInput(0).scalar_type());
  auto result = torch::_softmax_backward_data(
      GetHpuInput(0), GetHpuInput(1), {} /*dim*/, GetHpuInput(0).scalar_type());

  Compare(expected, result);
}

TEST_F(HpuOpTest, softmax_bwd_bf16) {
  GenerateInputs(2, torch::kBFloat16);
  auto expected = torch::_softmax_backward_data(
      GetCpuInput(0), GetCpuInput(1), 2 /*dim*/, GetCpuInput(0).scalar_type());
  auto result = torch::_softmax_backward_data(
      GetHpuInput(0), GetHpuInput(1), 2 /*dim*/, GetHpuInput(0).scalar_type());

  Compare(expected, result, 1e-03, 6e-02);
}

TEST_F(HpuOpTest, softmax_bwd_bf16_negdim) {
  GenerateInputs(2, torch::kBFloat16);
  auto expected = torch::_softmax_backward_data(
      GetCpuInput(0), GetCpuInput(1), -2 /*dim*/, GetCpuInput(0).scalar_type());
  auto result = torch::_softmax_backward_data(
      GetHpuInput(0), GetHpuInput(1), -2 /*dim*/, GetHpuInput(0).scalar_type());

  Compare(expected, result, 1e-03, 6e-02);
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
      GetCpuInput(1),
      GetCpuInput(2),
      /*dim*/ -1,
      GetCpuInput(0).scalar_type(),
      expected);
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
      GetCpuInput(1),
      GetCpuInput(2),
      /*dim*/ 3,
      GetCpuInput(0).scalar_type(),
      expected);
  torch::_softmax_backward_data_outf(
      hgrad_out, houtput, /*dim*/ 3, GetHpuInput(0).scalar_type(), result);

  Compare(expected, result, 1e-03, 6e-02);
}