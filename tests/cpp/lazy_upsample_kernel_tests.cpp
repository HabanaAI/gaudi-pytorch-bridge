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

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using namespace habana_lazy;
using namespace at;

class LazyUpsampleKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyUpsampleKernelTest, UpsampleNearestTest) {
  torch::Tensor tensor = torch::randn({3, 1, 5, 5});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  std::array<double, 2> scale_array = {2.0, 2.0};
  c10::ArrayRef<double> scale_factors = scale_array;
  auto outHabana = torch::upsample_nearest2d(tHabana, {}, scale_factors);
  auto out = torch::upsample_nearest2d(tensor, {}, scale_factors);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

/*
   This test uses cpu fallback
TEST_F(LazyUpsampleKernelTest, UpsampleNearest2Test) {
  torch::Tensor tensor = torch::randn({3, 1, 5, 5});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  std::array<int64_t, 2> sizes_array = {10, 15};
  c10::ArrayRef<int64_t> sizes = sizes_array;
  auto outHabana = torch::upsample_nearest2d(tHabana, sizes, {});
  auto out = torch::upsample_nearest2d(tensor, sizes, {});
  // EXPECT_DEATH(outHabana.to(torch::kCPU), "*");
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}
*/

TEST_F(LazyUpsampleKernelTest, UpsampleNearestTest_channelLast) {
  torch::Tensor tensor =
      torch::randn({2, 3, 4, 5}).to(c10::MemoryFormat::ChannelsLast);
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  std::array<double, 2> scale_array = {2.0, 2.0};
  c10::ArrayRef<double> scale_factors = scale_array;
  auto outHabana = torch::upsample_nearest2d(tHabana, {}, scale_factors);
  auto out = torch::upsample_nearest2d(tensor, {}, scale_factors);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyUpsampleKernelTest, UpsampleBackwardTest) {
  torch::manual_seed(0);
  auto upsample_test = [](c10::IntArrayRef size1) {
    auto mat1 = torch::randn(size1);
    auto mat1_h = mat1.to(torch::kHPU);
    mat1.set_requires_grad(true);

    auto out = torch::upsample_nearest2d(mat1, {8, 21});
    auto grad_out = torch::ones_like(out);
    auto grad_out_h = grad_out.to(torch::kHPU);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad();

    torch::Tensor grad_mat1_h;

    c10::optional<double> scales_h(2.0);
    c10::optional<double> scales_w(3.0);
    std::array<int64_t, 2> out_sizes = {8, 21};
    c10::IntArrayRef out_size = out_sizes;
    grad_mat1_h = torch::upsample_nearest2d_backward(
        grad_out_h, out_size, size1, scales_h, scales_w);

    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal1, true);
  };
  upsample_test({1, 1, 4, 7});
}

TEST_F(LazyUpsampleKernelTest, UpsampleBackwardTest_channelLast) {
  torch::manual_seed(0);
  auto upsample_test = [](c10::IntArrayRef size1) {
    auto mat1 = torch::randn(size1).to(c10::MemoryFormat::ChannelsLast);
    auto mat1_h = mat1.to(torch::kHPU);
    mat1.set_requires_grad(true);

    auto out = torch::upsample_nearest2d(mat1, {8, 21});
    auto grad_out = torch::ones_like(out);
    auto grad_out_h = grad_out.to(torch::kHPU);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad();

    torch::Tensor grad_mat1_h;

    c10::optional<double> scales_h(2.0);
    c10::optional<double> scales_w(3.0);
    std::array<int64_t, 2> out_sizes = {8, 21};
    c10::IntArrayRef out_size = out_sizes;
    grad_mat1_h = torch::upsample_nearest2d_backward(
        grad_out_h, out_size, size1, scales_h, scales_w);

    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal1, true);
  };
  upsample_test({1, 1, 4, 7});
}

TEST_F(LazyUpsampleKernelTest, DS_UpsampleBackwardTest) {
  torch::manual_seed(0);
  bool refine_enabled = habana_helpers::GetRefineDynamicShapeStatus();
  if (!refine_enabled) {
    habana_helpers::EnableRefineDynamicShape();
  }
  auto upsample_test = [](c10::IntArrayRef size1) {
    auto mat1 = torch::randn(size1);
    auto mat1_h = mat1.to(torch::kHPU);
    mat1.set_requires_grad(true);
    std::array<double, 2> scales = {2.0, 3.0};
    c10::optional<c10::ArrayRef<double>> scale_factors = scales;
    std::array<int64_t, 2> out_sizes = {8, 21};
    c10::optional<c10::IntArrayRef> out_size = c10::nullopt;

    auto out = torch::upsample_nearest2d(mat1, out_size, scale_factors);
    auto grad_out = torch::ones_like(out);
    auto grad_out_h = grad_out.to(torch::kHPU);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad();

    torch::Tensor grad_mat1_h;

    c10::IntArrayRef out_size2 = out_sizes;
    c10::optional<double> scales_h(2.0);
    c10::optional<double> scales_w(3.0);
    grad_mat1_h = torch::upsample_nearest2d_backward(
        grad_out_h, out_size2, size1, scales_h, scales_w);

    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal1, true);
  };
  upsample_test({1, 1, 2, 3});
  upsample_test({1, 1, 4, 7});
  upsample_test({1, 1, 6, 12});
  if (!refine_enabled) {
    habana_helpers::DisableRefineDynamicShape();
  }
}

TEST_F(LazyUpsampleKernelTest, UpsampleNearestTest3D) {
  torch::Tensor tensor = torch::randn({3, 6, 8, 4, 2});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  std::array<double, 3> scale_array = {1.0, 2.0, 3.0};
  c10::ArrayRef<double> scale_factors = scale_array;
  auto outHabana = torch::upsample_nearest3d(tHabana, {}, scale_factors);
  auto out = torch::upsample_nearest3d(tensor, {}, scale_factors);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyUpsampleKernelTest, UpsampleNearestTest3D_channelLast) {
  torch::Tensor tensor =
      torch::randn({5, 1, 7, 2, 3}).to(c10::MemoryFormat::ChannelsLast3d);
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  std::array<double, 3> scale_array = {2.0, 1.0, 4.0};
  c10::ArrayRef<double> scale_factors = scale_array;
  auto outHabana = torch::upsample_nearest3d(tHabana, {}, scale_factors);
  auto out = torch::upsample_nearest3d(tensor, {}, scale_factors);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}
