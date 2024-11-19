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

TEST_F(HpuOpTest, softplus_f32) {
  GenerateInputs(1, {{2, 3, 4}});

  float beta = 1.5f;
  float threshold = 20.0f;
  auto exp = torch::softplus(GetCpuInput(0), beta, threshold);
  auto res = torch::softplus(GetHpuInput(0), beta, threshold);

  Compare(exp, res);
}

TEST_F(HpuOpTest, softplus_bf16) {
  GenerateInputs(1, {{2, 3, 4}}, torch::kBFloat16);

  float beta = 1.5f;
  float threshold = 20.0f;
  auto exp = torch::softplus(GetCpuInput(0), beta, threshold);
  auto res = torch::softplus(GetHpuInput(0), beta, threshold);

  Compare(exp, res, 0.01, 0.01);
}

TEST_F(HpuOpTest, softplus_out_f32) {
  GenerateInputs(1, {{2, 3, 4}});

  float beta = 1.5f;
  float threshold = 20.0f;

  auto exp = torch::empty(0, torch::kFloat32);
  auto res = exp.to(torch::kHPU);

  torch::softplus_outf(GetCpuInput(0), beta, threshold, exp);
  torch::softplus_outf(GetHpuInput(0), beta, threshold, res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, softplus_out_bf16) {
  GenerateInputs(1, {{2, 3, 4}}, torch::kBFloat16);

  float beta = 1.5f;
  float threshold = 20.0f;

  auto exp = torch::empty(0, torch::kBFloat16);
  auto res = exp.to(torch::kHPU);

  torch::softplus_outf(GetCpuInput(0), beta, threshold, exp);
  torch::softplus_outf(GetHpuInput(0), beta, threshold, res);

  Compare(exp, res, 0.01, 0.01);
}
// Bwd

TEST_F(HpuOpTest, softplus_bwd_f32) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}});

  float beta = 1.5f;
  float threshold = 20.0f;
  auto exp = torch::softplus_backward(
      GetCpuInput(0) /*grad_output*/, GetCpuInput(1) /*self*/, beta, threshold);
  auto res = torch::softplus_backward(
      GetHpuInput(0) /*grad_output*/, GetCpuInput(1) /*self*/, beta, threshold);

  Compare(exp, res);
}

TEST_F(HpuOpTest, softplus_bwd_bf16) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}}, torch::kBFloat16);

  float beta = 1.5f;
  float threshold = 20.0f;
  auto exp = torch::softplus_backward(
      GetCpuInput(0) /*grad_output*/, GetCpuInput(1) /*self*/, beta, threshold);
  auto res = torch::softplus_backward(
      GetHpuInput(0) /*grad_output*/, GetCpuInput(1) /*self*/, beta, threshold);

  Compare(exp, res, 0.01, 0.01);
}

TEST_F(HpuOpTest, softplus_out_bwd_f32) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}});

  float beta = 1.5f;
  float threshold = 20.0f;

  auto exp = torch::empty(0, torch::kFloat32);
  auto res = exp.to(torch::kHPU);

  torch::softplus_backward_outf(
      GetCpuInput(0) /*grad_output*/,
      GetCpuInput(1) /*self*/,
      beta,
      threshold,
      exp);
  torch::softplus_backward_outf(
      GetHpuInput(0) /*grad_output*/,
      GetHpuInput(1) /*self*/,
      beta,
      threshold,
      res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, softplus_out_bwd_bf16) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}}, torch::kBFloat16);

  float beta = 1.5f;
  float threshold = 20.0f;

  auto exp = torch::empty(0, torch::kBFloat16);
  auto res = exp.to(torch::kHPU);

  torch::softplus_backward_outf(
      GetCpuInput(0) /*grad_output*/,
      GetCpuInput(1) /*self*/,
      beta,
      threshold,
      exp);
  torch::softplus_backward_outf(
      GetHpuInput(0) /*grad_output*/,
      GetHpuInput(1) /*self*/,
      beta,
      threshold,
      res);

  Compare(exp, res, 0.01, 0.01);
}