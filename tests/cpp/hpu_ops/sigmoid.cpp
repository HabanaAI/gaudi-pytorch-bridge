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

TEST_F(HpuOpTest, sigmoid) {
  GenerateInputs(1, {{1, 2}});

  auto expected = torch::sigmoid(GetCpuInput(0));
  auto result = torch::sigmoid(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, sigmoid_) {
  GenerateInputs(1, {{1, 2}});

  GetCpuInput(0).sigmoid_();
  GetHpuInput(0).sigmoid_();

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, sigmoid_out) {
  GenerateInputs(1, {{2, 3, 5}});

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sigmoid_outf(GetCpuInput(0), expected);
  torch::sigmoid_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sigmoid_backward_out) {
  GenerateInputs(1, {{2, 3, 5}});

  torch::Tensor grad_out = torch::ones({2, 3, 5});

  torch::ScalarType dtype = torch::kFloat;
  grad_out = grad_out.to(dtype);

  auto hgrad_out = grad_out.to(torch::kHPU, dtype);
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sigmoid_backward_outf(grad_out, GetCpuInput(0), expected);
  torch::sigmoid_backward_outf(hgrad_out, GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, hardsigmoid_bwd_out) {
  GenerateInputs(2);

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::hardsigmoid_backward_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::hardsigmoid_backward_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}