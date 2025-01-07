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

// Note: No need to compare second o/p since it contains results of
// computations CPU does to use in the CPU backward pass impl.
// This is non standard. So do no compare this tensor.
TEST_F(HpuOpTest, log_sigmoid_fwd) {
  GenerateInputs(1);

  auto exp = torch::log_sigmoid_forward(GetCpuInput(0));
  auto res = torch::log_sigmoid_forward(GetHpuInput(0));
  Compare(std::get<0>(exp), std::get<0>(res));
  // No need to compare second o/p. See Note above.
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
  // No need to compare second o/p(i.e, buffer). See Note above.
}

TEST_F(HpuOpTest, log_sigmoid_bwd) {
  GenerateInputs(1);
  at::Tensor cpu_fwd_result, hpu_fwd_result, cpu_buffer, hpu_buffer;
  auto cpu_input = GetCpuInput(0);
  auto hpu_input = GetHpuInput(0);
  std::tie(cpu_fwd_result, cpu_buffer) = torch::log_sigmoid_forward(cpu_input);
  std::tie(hpu_fwd_result, hpu_buffer) = torch::log_sigmoid_forward(hpu_input);
  auto cpu_grad = torch::ones_like(cpu_fwd_result);
  auto hpu_grad = torch::ones_like(hpu_fwd_result);
  auto expected = torch::log_sigmoid_backward(cpu_grad, cpu_input, cpu_buffer);
  auto result = torch::log_sigmoid_backward(hpu_grad, hpu_input, hpu_buffer);
  Compare(expected, result);
}

TEST_F(HpuOpTest, log_sigmoid_bwd_out) {
  GenerateInputs(1);

  torch::ScalarType dtype = torch::kFloat;
  at::Tensor cpu_fwd_result, hpu_fwd_result, cpu_buffer, hpu_buffer;
  auto cpu_input = GetCpuInput(0);
  auto hpu_input = GetHpuInput(0);
  std::tie(cpu_fwd_result, cpu_buffer) = torch::log_sigmoid_forward(cpu_input);
  std::tie(hpu_fwd_result, hpu_buffer) = torch::log_sigmoid_forward(hpu_input);
  auto cpu_grad = torch::ones_like(cpu_fwd_result);
  auto hpu_grad = torch::ones_like(hpu_fwd_result);
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::log_sigmoid_backward_outf(cpu_grad, cpu_input, cpu_buffer, expected);
  torch::log_sigmoid_backward_outf(hpu_grad, hpu_input, hpu_buffer, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, log_sigmoid) {
  GenerateInputs(1);
  auto exp = torch::log_sigmoid(GetCpuInput(0));
  auto res = torch::log_sigmoid(GetHpuInput(0));
  Compare(exp, res);
}

TEST_F(HpuOpTest, log_sigmoid_out) {
  GenerateInputs(1);

  torch::ScalarType dtype = torch::kFloat;
  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::log_sigmoid_outf(GetCpuInput(0), exp);
  torch::log_sigmoid_outf(GetHpuInput(0), res);
  Compare(exp, res);
}
