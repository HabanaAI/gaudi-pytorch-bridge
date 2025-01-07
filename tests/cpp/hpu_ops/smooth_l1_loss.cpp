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

class HpuOpTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<std::tuple<double, int64_t>> {};

TEST_P(HpuOpTest, smooth_l1_loss) {
  GenerateInputs(2, {{2, 3, 5}, {2, 3, 5}});
  const auto& testParams = GetParam();
  const auto beta = std::get<0>(testParams);
  const auto mode = std::get<1>(testParams);

  auto expected =
      torch::smooth_l1_loss(GetCpuInput(0), GetCpuInput(1), mode, beta);
  auto result =
      torch::smooth_l1_loss(GetHpuInput(0), GetHpuInput(1), mode, beta);
  Compare(expected, result);
}

TEST_P(HpuOpTest, smooth_l1_loss_out) {
  GenerateInputs(2, {{3, 3, 5}, {3, 3, 5}});
  const auto& testParams = GetParam();
  const auto beta = std::get<0>(testParams);
  const auto mode = std::get<1>(testParams);

  auto result =
      torch::empty({}, torch::TensorOptions(torch::kFloat).device("hpu"));

  // Not Using outf variant for CPU as it gives incorrest results
  auto expected =
      torch::smooth_l1_loss(GetCpuInput(0), GetCpuInput(1), mode, beta);
  torch::smooth_l1_loss_outf(
      GetHpuInput(0), GetHpuInput(1), mode, beta, result);
  Compare(expected, result);
}

TEST_P(HpuOpTest, smooth_l1_loss_backward) {
  GenerateInputs(2, {{2, 3, 5}, {2, 3, 5}});
  const auto& testParams = GetParam();
  const auto beta = std::get<0>(testParams);
  const auto mode = std::get<1>(testParams);

  torch::Tensor grad_out = torch::ones({1});
  if (mode == at::Reduction::None) {
    grad_out = torch::ones({2, 3, 5});
  }
  torch::ScalarType dtype = torch::kFloat;
  grad_out = grad_out.to(dtype);
  auto hgrad_out = grad_out.to(torch::kHPU, dtype);

  auto expected = torch::smooth_l1_loss_backward(
      grad_out, GetCpuInput(0), GetCpuInput(1), mode, beta);
  auto result = torch::smooth_l1_loss_backward(
      hgrad_out, GetHpuInput(0), GetHpuInput(1), mode, beta);
  Compare(expected, result);
}

TEST_P(HpuOpTest, smooth_l1_loss_backward_out) {
  GenerateInputs(2, {{2, 3, 5}, {2, 3, 5}});
  const auto& testParams = GetParam();
  const auto beta = std::get<0>(testParams);
  const auto mode = std::get<1>(testParams);

  torch::Tensor grad_out = torch::ones({1});
  if (mode == at::Reduction::None) {
    grad_out = torch::ones({2, 3, 5});
  }

  torch::ScalarType dtype = torch::kFloat;
  grad_out = grad_out.to(dtype);
  auto hgrad_out = grad_out.to(torch::kHPU, dtype);

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::smooth_l1_loss_backward_outf(
      grad_out, GetCpuInput(0), GetCpuInput(1), mode, beta, expected);
  torch::smooth_l1_loss_backward_outf(
      hgrad_out, GetHpuInput(0), GetHpuInput(1), mode, beta, result);
  Compare(expected, result);
}

INSTANTIATE_TEST_SUITE_P(
    sanity,
    HpuOpTest,
    ::testing::Combine(
        ::testing::Values<double>(0, .5, .7, 1.0, 1.5, 2.5),
        ::testing::Values<int64_t>(
            at::Reduction::None,
            at::Reduction::Mean,
            at::Reduction::Sum)));
