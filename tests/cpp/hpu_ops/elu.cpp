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
