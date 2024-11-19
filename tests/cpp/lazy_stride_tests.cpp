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

#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;

class LazyStridesTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyStridesTest, NCHWInputTensorsStrides) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int BATCH = 16;
  const int H = 64, W = 64;

  std::cout << '\n';
  std::cout << "PTI_DBG :: NCHWInputTensorsStrides Test --------" << '\n';
  // weight_tensor = bias1 + bias2
  torch::Tensor bias1 =
      torch::randn({C, C, kW, kH}, torch::requires_grad(false));
  torch::Tensor bias2 =
      torch::randn({C, C, kW, kH}, torch::requires_grad(false));
  torch::Tensor h_bias1 = bias1.to(torch::kHPU);
  torch::Tensor h_bias2 = bias2.to(torch::kHPU);

  torch::Tensor weight_tensor = torch::add(bias1, bias2);
  torch::Tensor h_weight_tensor = torch::add(h_bias1, h_bias2);

  std::cout << "PTI_DBG ::"
            << " weight_tensor.shape : " << weight_tensor.sizes()
            << " weight_tensor.strides : " << weight_tensor.strides() << '\n';
  std::cout << "PTI_DBG ::"
            << " h_weight_tensor.shape : " << h_weight_tensor.sizes()
            << " h_weight_tensor.strides : " << h_weight_tensor.strides()
            << '\n';

  // out_conv = Conv3x3(Data, weight)
  torch::Tensor in_tensor =
      torch::randn({BATCH, C, H, W}, torch::requires_grad(false));
  torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
  torch::Tensor h_weight_tensor_hwck = h_weight_tensor;

  torch::Tensor h_out_conv = torch::conv2d(
      h_in_tensor, h_weight_tensor_hwck, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::Tensor out_conv = torch::conv2d(
      in_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::Tensor out_conv_hpu = h_out_conv.to(torch::kCPU);

  std::cout << "PTI_DBG ::"
            << " out_conv.shape : " << out_conv.sizes()
            << " out_conv.strides : " << out_conv.strides() << '\n';
  std::cout << "PTI_DBG ::"
            << " out_conv_hpu.shape : " << out_conv_hpu.sizes()
            << " out_conv_hpu.strides : " << out_conv_hpu.strides() << '\n';

  EXPECT_EQ(out_conv.sizes() == out_conv_hpu.sizes(), true);
  EXPECT_EQ(out_conv.strides() == out_conv_hpu.strides(), true);
}

TEST_F(LazyStridesTest, SimpleStrideTest) {
  const int A = 4;
  const int B = 3;
  const int C = 3;

  std::cout << '\n';
  std::cout << "PTI_DBG :: SimpleStrideTest Test --------" << '\n';
  torch::Tensor c0 = torch::randn({C, B, A}, torch::requires_grad(false));
  torch::Tensor c1 = torch::randn({C, B, A}, torch::requires_grad(false));

  torch::Tensor c4 = torch::add(c0, c1);
  torch::Tensor c5 = torch::mul(c0, c1);
  torch::Tensor c6 = torch::mul(c4, c5);
  torch::Tensor c7 = torch::relu(c6);

  std::cout << "PTI_DBG ::"
            << " c0.shape : " << c0.sizes() << " c0.strides : " << c0.strides()
            << '\n';
  std::cout << "PTI_DBG ::"
            << " c1.shape : " << c1.sizes() << " c1.strides : " << c1.strides()
            << '\n';

  std::cout << "PTI_DBG ::"
            << " c7.shape : " << c7.sizes() << " c7.strides : " << c7.strides()
            << '\n';

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = c1.to(torch::kHPU);
  torch::Tensor h4 = torch::add(h0, h1);
  torch::Tensor h5 = torch::mul(h0, h1);
  torch::Tensor h6 = torch::mul(h4, h5);
  torch::Tensor h7 = torch::relu(h6);
  torch::Tensor h7_c = h7.to(torch::kCPU);

  std::cout << "PTI_DBG ::"
            << " h0.shape : " << h0.sizes() << " h0.strides : " << h0.strides()
            << '\n';
  std::cout << "PTI_DBG ::"
            << " h1.shape : " << h1.sizes() << " h1.strides : " << h1.strides()
            << '\n';

  std::cout << "PTI_DBG ::"
            << " h7.shape : " << h7.sizes() << " h7.strides : " << h7.strides()
            << '\n';

  EXPECT_EQ(c7.sizes() == h7_c.sizes(), true);
  EXPECT_EQ(c7.strides() == h7_c.strides(), true);
}

TEST_F(LazyStridesTest, NonContigiousStrides) {
  const int A = 2;
  const int B = 3;
  const int C = 4;

  std::cout << '\n';
  std::cout << "PTI_DBG :: NonContigiousStrides Test --------" << '\n';
  torch::Tensor c0 = torch::randn({A, B, C}, torch::requires_grad(false));

  torch::Tensor c1 = c0.transpose(1, 0);

  std::cout << "PTI_DBG ::"
            << " c0.shape : " << c0.sizes() << " c0.strides : " << c0.strides()
            << '\n';
  std::cout << "PTI_DBG ::"
            << " c1.shape : " << c1.sizes() << " c1.strides : " << c1.strides()
            << '\n';

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = h0.transpose(1, 0);
  torch::Tensor h1_c = h1.to(torch::kCPU);

  std::cout << "PTI_DBG ::"
            << " h0.shape : " << h0.sizes() << " h0.strides : " << h0.strides()
            << '\n';
  std::cout << "PTI_DBG ::"
            << " h1.shape : " << h1.sizes() << " h1.strides : " << h1.strides()
            << '\n';

  std::cout << "PTI_DBG ::"
            << " h1_c.shape : " << h1_c.sizes() << '\n';

  EXPECT_EQ(c1.sizes() == h1_c.sizes(), true);
  EXPECT_EQ(c1.sizes() == h1.sizes(), true);
  // Comparing c1 and h1 strides, beacuse h1_c strides might not match as tensor
  // storage is managed differently.
  EXPECT_EQ(c1.strides() == h1.strides(), true);
}

TEST_F(LazyStridesTest, ZeroElementStrides) {
  const int A = 2;
  const int B = 0;
  const int C = 4;

  std::cout << '\n';
  std::cout << "PTI_DBG :: ZeroElementStrides Test --------" << '\n';
  torch::Tensor c0 = torch::randn({A, B, C}, torch::requires_grad(false));

  torch::Tensor c1 = c0.abs();

  std::cout << "PTI_DBG ::"
            << " c0.shape : " << c0.sizes() << " c0.strides : " << c0.strides()
            << '\n';
  std::cout << "PTI_DBG ::"
            << " c1.shape : " << c1.sizes() << " c1.strides : " << c1.strides()
            << '\n';

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = h0.abs();
  torch::Tensor h1_c = h1.to(torch::kCPU);

  std::cout << "PTI_DBG ::"
            << " h0.shape : " << h0.sizes() << " h0.strides : " << h0.strides()
            << '\n';
  std::cout << "PTI_DBG ::"
            << " h1.shape : " << h1.sizes() << " h1.strides : " << h1.strides()
            << '\n';

  std::cout << "PTI_DBG ::"
            << " h1_c.shape : " << h1_c.sizes()
            << " h1_c.strides : " << h1_c.strides() << '\n';

  EXPECT_EQ(c1.sizes() == h1_c.sizes(), true);
  EXPECT_EQ(c1.strides() == h1_c.strides(), true);
}
