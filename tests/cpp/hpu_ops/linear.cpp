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

TEST_F(HpuOpTest, linear_3d) {
  GenerateInputs(2, {{8, 4, 12, 7}, {5, 7}}, {torch::kBFloat16});

  auto expected = torch::linear(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::linear(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}
TEST_F(HpuOpTest, linear_4d) {
  GenerateInputs(2, {{2, 4, 5, 7, 9}, {3, 9}});

  auto expected = torch::linear(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::linear(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

auto linear_test = [](std::vector<int64_t> in_shape /* shape upto n-1*/,
                      bool bias_required,
                      bool dynamic) {
  int out_features = 5;
  int in_features = 4;

  for (int i = 0; i <= 2 * dynamic; i++) {
    out_features += i;
    in_features += i;
    in_shape.push_back(in_features);

    auto in = torch::randn(in_shape, torch::requires_grad());
    auto hin = in.to(torch::kHPU).detach().requires_grad_();
    auto wt = torch::randn(
        {out_features, in_features}, torch::requires_grad()); // ckhw
    auto hwt = wt.to(torch::kHPU).detach().requires_grad_();
    auto bias = torch::randn({out_features}, torch::requires_grad());
    auto hbias = bias.to(torch::kHPU).detach().requires_grad_();
    if (!bias_required) {
      bias = torch::Tensor();
      hbias = torch::Tensor();
    }
    auto exp = torch::linear(in, wt, bias);
    auto exp_hpu = torch::linear(hin, hwt, hbias);

    auto grad_out = torch::ones_like(exp.detach());
    auto hgrad_out = grad_out.detach().to(torch::kHPU);

    exp.backward(grad_out);
    exp_hpu.backward(hgrad_out);

    auto grad_in = in.grad();
    auto grad_wt = wt.grad();
    at::Tensor grad_bias;
    if (bias_required)
      grad_bias = bias.grad();

    at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
    hgrad_in = hin.grad();
    hgrad_wt = hwt.grad();
    if (bias_required)
      hgrad_bias = hbias.grad();

    auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
    auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);

    at::Tensor hgrad_bias_cpu;
    if (bias_required)
      hgrad_bias_cpu = hgrad_bias.to(torch::kCPU);

    EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
    EXPECT_EQ(allclose(grad_in, hgrad_in_cpu, 0.01, 0.01), true);
    if (bias_required)
      EXPECT_EQ(allclose(grad_bias, hgrad_bias_cpu, 0.01, 0.01), true);
    in_shape.pop_back();
  }
};

auto linear_backward_test =
    [](std::vector<int64_t> in_shape /* shape upto n-1*/,
       bool bias_required,
       bool dynamic) {
      int out_features = 5;
      int in_features = 4;

      for (int i = 0; i <= 2 * dynamic; i++) {
        out_features += i;
        in_features += i;
        in_shape.push_back(in_features);

        auto in = torch::randn(in_shape, torch::requires_grad());
        auto hin = in.to(torch::kHPU).detach().requires_grad_();
        auto wt = torch::randn(
            {out_features, in_features}, torch::requires_grad()); // ckhw
        auto hwt = wt.to(torch::kHPU).detach().requires_grad_();
        auto bias = torch::randn({out_features}, torch::requires_grad());
        auto hbias = bias.to(torch::kHPU).detach().requires_grad_();
        if (!bias_required) {
          bias = torch::Tensor();
          hbias = torch::Tensor();
        }
        auto exp = torch::linear(in, wt, bias);
        auto exp_hpu = torch::linear(hin, hwt, hbias);

        auto grad_out = torch::ones_like(exp.detach());
        auto hgrad_out = grad_out.detach().to(torch::kHPU);

        exp.backward(grad_out);

        auto grad_in = in.grad();
        auto grad_wt = wt.grad();
        at::Tensor grad_bias;
        if (bias_required)
          grad_bias = bias.grad();

        at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
        hgrad_in = hin.grad();
        hgrad_wt = hwt.grad();
        if (bias_required)
          hgrad_bias = hbias.grad();
        std::array<bool, 3> mask{1, 1, bias_required};

        std::tie(hgrad_in, hgrad_wt, hgrad_bias) =
            linear_backward(hin, hgrad_out, hwt, mask);

        auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
        auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
        at::Tensor hgrad_bias_cpu;
        if (bias_required)
          hgrad_bias_cpu = hgrad_bias.to(torch::kCPU);

        EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
        EXPECT_EQ(allclose(grad_in, hgrad_in_cpu, 0.01, 0.01), true);
        if (bias_required)
          EXPECT_EQ(allclose(grad_bias, hgrad_bias_cpu, 0.01, 0.01), true);

        in_shape.pop_back();
      }
    };

TEST_F(HpuOpTest, LinearBwdTest2D) {
  linear_test({3}, 0, 0);
}
TEST_F(HpuOpTest, LinearBwdTest2DBias) {
  linear_test({3}, 1, 0);
}
TEST_F(HpuOpTest, LinearBwdTest2DDynamic) {
  linear_test({3}, 0, 1);
}
TEST_F(HpuOpTest, LinearBwdTest2DBiasDynamic) {
  linear_test({3}, 1, 1);
}

TEST_F(HpuOpTest, LinearBwdTest1D) {
  linear_test({}, 0, 0);
}
TEST_F(HpuOpTest, LinearBwdTest1DBias) {
  linear_test({}, 1, 0);
}
TEST_F(HpuOpTest, LinearBwdTest1DDynamic) {
  linear_test({}, 0, 1);
}
TEST_F(HpuOpTest, LinearBwdTest1DBiasDynamic) {
  linear_test({}, 1, 1);
}

TEST_F(HpuOpTest, LinearBwdTest3D) {
  linear_test({2, 3}, 0, 0);
}
TEST_F(HpuOpTest, LinearBwdTest3DBias) {
  linear_test({2, 3}, 1, 0);
}
TEST_F(HpuOpTest, LinearBwdTest3DDynamic) {
  linear_test({2, 3}, 0, 1);
}
TEST_F(HpuOpTest, LinearBwdTest3DBiasDynamic) {
  linear_test({2, 3}, 1, 1);
}

TEST_F(HpuOpTest, LinearBwdTest4D) {
  linear_test({2, 4, 3}, 0, 0);
}
TEST_F(HpuOpTest, LinearBwdTest4DBias) {
  linear_test({2, 4, 3}, 1, 0);
}
TEST_F(HpuOpTest, LinearBwdTest4DDynamic) {
  linear_test({2, 4, 3}, 0, 1);
}
TEST_F(HpuOpTest, LinearBwdTest4DBiasDynamic) {
  linear_test({2, 4, 3}, 1, 1);
}

// aten::linear_backward tests
TEST_F(HpuOpTest, LinearBackwardTest2D) {
  linear_backward_test({3}, 0, 0);
}
TEST_F(HpuOpTest, LinearBackwardTest2DBias) {
  linear_backward_test({3}, 1, 0);
}
TEST_F(HpuOpTest, LinearBackwardTest2DDynamic) {
  linear_backward_test({3}, 0, 1);
}
TEST_F(HpuOpTest, LinearBackwardTest2DBiasDynamic) {
  linear_backward_test({3}, 1, 1);
}

TEST_F(HpuOpTest, LinearBackwardTest1D) {
  linear_backward_test({}, 0, 0);
}
TEST_F(HpuOpTest, LinearBackwardTest1DBias) {
  linear_backward_test({}, 1, 0);
}
TEST_F(HpuOpTest, LinearBackwardTest1DDynamic) {
  linear_backward_test({}, 0, 1);
}
TEST_F(HpuOpTest, LinearBackwardTest1DBiasDynamic) {
  linear_backward_test({}, 1, 1);
}

TEST_F(HpuOpTest, LinearBackwardTest3D) {
  linear_backward_test({2, 3}, 0, 0);
}
TEST_F(HpuOpTest, LinearBackwardTest3DBias) {
  linear_backward_test({2, 3}, 1, 0);
}
TEST_F(HpuOpTest, LinearBackwardTest3DDynamic) {
  linear_backward_test({2, 3}, 0, 1);
}
TEST_F(HpuOpTest, LinearBackwardTest3DBiasDynamic) {
  linear_backward_test({2, 3}, 1, 1);
}

TEST_F(HpuOpTest, LinearBackwardTest4D) {
  linear_backward_test({2, 4, 3}, 0, 0);
}
TEST_F(HpuOpTest, LinearBackwardTest4DBias) {
  linear_backward_test({2, 4, 3}, 1, 0);
}
TEST_F(HpuOpTest, LinearBackwardTest4DDynamic) {
  linear_backward_test({2, 4, 3}, 0, 1);
}
TEST_F(HpuOpTest, LinearBackwardTest4DBiasDynamic) {
  linear_backward_test({2, 4, 3}, 1, 1);
}
