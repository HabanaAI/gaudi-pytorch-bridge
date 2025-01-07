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

#include "habana_kernels/lazy_kernels_declarations.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, rrelu_with_noise) {
  GenerateInputs(2, {torch::kBFloat16});
  float lower = 0.1;
  float upper = 0.3;
  bool training = false;
  SetSeed();
  auto expected = torch::rrelu_with_noise(
      GetCpuInput(0),
      GetCpuInput(1),
      lower,
      upper,
      training,
      at::detail::getDefaultCPUGenerator());
  SetSeed();
  auto result = torch::rrelu_with_noise(
      GetHpuInput(0),
      GetHpuInput(1),
      lower,
      upper,
      training,
      at::detail::getDefaultCPUGenerator());
  double atol = 1.7e-3;
  double rtol = 1e-3;
  Compare(expected, result, rtol, atol);
}

TEST_F(HpuOpTest, rrelu_with_noise_train) {
  GenerateInputs(2);
  float lower = GenerateScalar<float>(0.1, 0.3);
  float upper = GenerateScalar<float>(0.6, 0.9);
  bool training = true;
  auto gen1 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto gen2 = at::detail::createCPUGenerator(/*seed_val=*/67280421310721);
  auto expected = torch::rrelu_with_noise(
      GetHpuInput(0), GetHpuInput(1), lower, upper, training, gen1);
  auto result = torch::rrelu_with_noise(
      GetHpuInput(0), GetHpuInput(1), lower, upper, training, gen2);
  EXPECT_TRUE(expected.equal(result));
}

TEST_F(HpuOpTest, rrelu_with_noise_default) {
  GenerateInputs(2);
  float lower = GenerateScalar<float>(0.1, 0.3);
  float upper = GenerateScalar<float>(0.6, 0.9);
  bool training = false;
  SetSeed();
  auto expected = torch::rrelu_with_noise(
      GetCpuInput(0), GetCpuInput(1), lower, upper, training);
  SetSeed();
  auto result = torch::rrelu_with_noise(
      GetHpuInput(0), GetHpuInput(1), lower, upper, training);
  Compare(expected, result);
}

TEST_F(HpuOpTest, rrelu_with_noise_inplace) {
  GenerateInputs(2);
  float lower = GenerateScalar<float>(0.1, 0.3);
  float upper = GenerateScalar<float>(0.6, 0.9);
  bool training = false;
  SetSeed();
  torch::rrelu_with_noise_(
      GetCpuInput(0),
      GetCpuInput(1),
      lower,
      upper,
      training,
      at::detail::getDefaultCPUGenerator());
  SetSeed();
  torch::rrelu_with_noise_(
      GetHpuInput(0),
      GetHpuInput(1),
      lower,
      upper,
      training,
      at::detail::getDefaultCPUGenerator());
  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, rrelu_with_noise_out) {
  GenerateInputs(2);
  float lower = GenerateScalar<float>(0.1, 0.3);
  float upper = GenerateScalar<float>(0.6, 0.9);
  bool training = false;
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::rrelu_with_noise_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      lower,
      upper,
      training,
      at::detail::getDefaultCPUGenerator(),
      expected);
  torch::rrelu_with_noise_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      lower,
      upper,
      training,
      at::detail::getDefaultCPUGenerator(),
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, rrelu_with_noise_backward) {
  GenerateInputs(3, {torch::kBFloat16});
  float lower = 0.1;
  float upper = 0.3;
  bool training = false;
  auto expected = torch::rrelu_with_noise_backward(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      lower,
      upper,
      training,
      false);
  auto result = torch::rrelu_with_noise_backward(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      lower,
      upper,
      training,
      false);
  double atol = 1.7e-3;
  double rtol = 1e-3;
  Compare(expected, result, rtol, atol);
}

TEST_F(HpuOpTest, rrelu_with_noise_backward_train) {
  GenerateInputs(3);
  float lower = GenerateScalar<float>(0.1, 0.3);
  float upper = GenerateScalar<float>(0.6, 0.9);
  bool training = true;
  auto expected = torch::rrelu_with_noise_backward(
      GetCpuInput(0),
      GetCpuInput(1),
      GetCpuInput(2),
      lower,
      upper,
      training,
      false);
  auto result = torch::rrelu_with_noise_backward(
      GetHpuInput(0),
      GetHpuInput(1),
      GetHpuInput(2),
      lower,
      upper,
      training,
      false);
  Compare(expected, result);
}
