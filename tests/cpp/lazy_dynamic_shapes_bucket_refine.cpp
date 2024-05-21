/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "habana_lazy_test_infra.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_kernels/lazy_kernels_declarations.h"

#include "backend/helpers/dynamic_bucket_info.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using namespace habana_lazy;

// In this class both the pass fallback and compilation fallback are disabled
class LazyDynamicShapesBucketRefineTest
    : public habana_lazy_test::LazyDynamicTest {
 public:
  void enable_bucket_refinement() {
    if (false == GET_ENV_FLAG_NEW(PT_HPU_ENABLE_COMPILE_THREAD))
      SET_ENV_FLAG_NEW(PT_HPU_ENABLE_COMPILE_THREAD, true, 1);

    if (false == GET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE))
      SET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE, true, 1);
  }

  void disable_bucket_refinement() {
    UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_COMPILE_THREAD);
    UNSET_ENV_FLAG_NEW(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE);
  }
};

TEST_F(LazyDynamicShapesBucketRefineTest, RefineAddMulRelu) {
  enable_bucket_refinement();
  int A = 50;
  const int C = 30;

  std::vector<int> input_sizes{34, 16, 32, 22, 17, 18, 16};
  std::vector<int> test_rounds{1, 1, 1, 1, 1, 2, 50};

  int num;

  for (int i = 0; i < input_sizes.size(); i++) {
    for (int j = 1; j <= test_rounds[i]; j++) {
      int B = input_sizes[i];
      PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i + 1, ", round ", j, "  START");

      torch::Tensor h0 =
          torch::randn({C, B, A}, torch::requires_grad(false)).to(torch::kHPU);
      torch::Tensor h1 =
          torch::randn({C, B, A}, torch::requires_grad(false)).to(torch::kHPU);

      torch::Tensor h4 = torch::add(h0, h1);
      torch::Tensor h5 = torch::mul(h0, h1);
      torch::Tensor h6 = torch::mul(h4, h5);
      torch::Tensor h7 = torch::relu(h6);
      HbLazyTensor::StepMarker({});
      torch::Tensor h7_c = h7.to(torch::kCPU);

      PT_TEST_DEBUG("PTI_DBG :: TEST ", i + 1, ", round ", j, "  END");
      habana_helpers::DynamicBucketInfo::DumpDynamicRecipeStat();
      if (j > 30) {
        habana_helpers::DynamicBucketInfo::DisableBucketRefinement();
      }
    }
  }
  disable_bucket_refinement();
}

TEST_F(LazyDynamicShapesBucketRefineTest, DISABLED_RefineAddMulReluBig) {
  enable_bucket_refinement();
  int A = 50;
  const int C = 30;

  std::vector<int> input_sizes{34, 16, 32, 22, 17, 18, 16};
  std::vector<int> test_rounds{1, 1, 1, 1, 1, 2, 50};

  int num;
  int level_cnt = 4;

  for (int i = 0; i < input_sizes.size(); i++) {
    for (int j = 1; j <= test_rounds[i]; j++) {
      int B = input_sizes[i] * 10;
      PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i + 1, ", round ", j, "  START");

      torch::Tensor a0 =
          torch::randn({C, B, A}, torch::requires_grad(false)).to(torch::kHPU);
      torch::Tensor m0 =
          torch::randn({C, B, A}, torch::requires_grad(false)).to(torch::kHPU);

      torch::Tensor a1, m1;
      for (size_t l = 0; l < level_cnt; l++) {
        a1 = torch::add(a0, m0);
        m1 = torch::mul(a0, a1);
        a0 = a1;
        m0 = m1;
      }
      torch::Tensor ml0 = torch::mul(a0, m0);
      torch::Tensor rl0 = torch::relu(ml0);
      HbLazyTensor::StepMarker({});

      PT_TEST_DEBUG("PTI_DBG :: TEST ", i + 1, ", round ", j, "  END");
      habana_helpers::DynamicBucketInfo::DumpDynamicRecipeStat();
      if (j > 30) {
        habana_helpers::DynamicBucketInfo::DisableBucketRefinement();
      }
    }
  }
  disable_bucket_refinement();
}

TEST_F(LazyDynamicShapesBucketRefineTest, DISABLED_RefineWithMatmul) {
  enable_bucket_refinement();
  int level_cnt = 4;
  int A = 50;
  std::vector<int> input_sizes{340, 160, 320, 220, 170, 180, 160};
  std::vector<int> test_rounds{1, 1, 1, 1, 1, 2, 50};
  for (int i = 0; i < input_sizes.size(); i++) {
    for (int j = 1; j <= test_rounds[i]; j++) {
      int B = input_sizes[i];
      PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i + 1, ", round ", j, "  START");
      torch::Tensor c0 = torch::randn({A, B, B}, torch::requires_grad(false));
      torch::Tensor c1 = torch::randn({A, B, B}, torch::requires_grad(false));

      torch::Tensor a0 = c0.to(torch::kHPU);
      torch::Tensor m0 = c1.to(torch::kHPU);
      torch::Tensor a1, m1;
      for (size_t l = 0; l < level_cnt; l++) {
        a1 = torch::add(a0, m0);
        m1 = torch::matmul(a0, a1);
        a0 = a1;
        m0 = m1;
      }
      torch::Tensor ml0 = torch::mul(a0, m0);
      torch::Tensor rl0 = torch::relu(ml0);
      HbLazyTensor::StepMarker({});

      PT_TEST_DEBUG(
          "PTI_DBG :: a0.shape : ", a0.sizes(), " a0.strides : ", a0.strides());
      PT_TEST_DEBUG(
          "PTI_DBG :: m0.shape : ", m0.sizes(), " m0.strides : ", m0.strides());
      PT_TEST_DEBUG(
          "PTI_DBG :: rl0.shape : ",
          rl0.sizes(),
          " rl0.strides : ",
          rl0.strides());
      PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i + 1, ", round ", j, "  START");
    }
  }
  disable_bucket_refinement();
}

TEST_F(LazyDynamicShapesBucketRefineTest, DISABLED_RefineUpsamplingNearest2d) {
  enable_bucket_refinement();

  int count = -1;
  auto upsample_test = [&count](c10::IntArrayRef in_sizes) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", ++count, " ----\n");

    torch::Tensor tensor = torch::randn(in_sizes);
    torch::Tensor tHabana = tensor.to(torch::kHPU);
    tensor.set_requires_grad(true);
    std::array<double, 2> scale_array = {2.0, 2.0};
    c10::ArrayRef<double> scale_factors = scale_array;
    c10::optional<c10::IntArrayRef> out_size = c10::nullopt;

    auto outHabana =
        torch::upsample_nearest2d(tHabana, out_size, scale_factors);
    auto out = torch::upsample_nearest2d(tensor, out_size, scale_factors);
    auto grad_out = torch::ones_like(out);
    auto grad_out_h = grad_out.to(torch::kHPU);
    HbLazyTensor::StepMarker({});
    out.backward(grad_out);
    auto grad_mat1 = tensor.grad();
    torch::Tensor grad_mat1_h;

    std::array<int64_t, 2> out_sizes_arr = {8, 21};
    c10::IntArrayRef out_sizes = out_sizes_arr;
    c10::optional<double> scales_h(2.0);
    c10::optional<double> scales_w(2.0);

    grad_mat1_h = torch::upsample_nearest2d_backward(
        grad_out_h, out_sizes, in_sizes, scales_h, scales_w);

    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    HbLazyTensor::StepMarker({});

    bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
    EXPECT_EQ(equal, true);

    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", count, " ----\n");
  };

  upsample_test({2, 211, 127, 71});
  upsample_test({2, 256, 156, 100});
  upsample_test({2, 236, 156, 71});
  upsample_test({2, 248, 148, 82});
  upsample_test({2, 217, 155, 98});
  upsample_test({2, 231, 129, 97});
  upsample_test({2, 214, 135, 81});
  upsample_test({2, 247, 135, 78});
  upsample_test({2, 218, 142, 77});
  upsample_test({2, 243, 151, 80});
  upsample_test({2, 229, 128, 88});
  upsample_test({2, 219, 136, 82});
  upsample_test({2, 247, 132, 84});
  upsample_test({2, 255, 145, 71});
  upsample_test({2, 240, 143, 72});
  upsample_test({2, 245, 152, 87});
  upsample_test({2, 227, 143, 73});
  upsample_test({2, 215, 142, 73});
  upsample_test({2, 236, 132, 96});
  upsample_test({2, 230, 138, 95});
  upsample_test({2, 211, 144, 98});
  upsample_test({2, 224, 128, 98});
  upsample_test({2, 238, 138, 95});
  upsample_test({2, 241, 155, 77});
  upsample_test({2, 239, 150, 87});
  upsample_test({2, 221, 143, 71});
  upsample_test({2, 242, 134, 96});
  upsample_test({2, 255, 153, 91});
  upsample_test({2, 255, 141, 96});
  upsample_test({2, 243, 131, 91});
  upsample_test({2, 255, 134, 90});
  upsample_test({2, 224, 141, 77});
  upsample_test({2, 241, 148, 78});
  upsample_test({2, 227, 140, 73});
  upsample_test({2, 212, 136, 93});
  upsample_test({2, 213, 146, 91});
  upsample_test({2, 242, 140, 85});
  upsample_test({2, 217, 138, 78});
  upsample_test({2, 212, 147, 83});
  upsample_test({2, 233, 129, 99});
  upsample_test({2, 233, 138, 76});
  upsample_test({2, 242, 127, 94});
  upsample_test({2, 235, 128, 83});
  upsample_test({2, 237, 138, 94});
  upsample_test({2, 212, 140, 87});
  upsample_test({2, 238, 138, 99});
  upsample_test({2, 220, 145, 100});
  upsample_test({2, 237, 137, 99});
  upsample_test({2, 226, 143, 83});
  upsample_test({2, 246, 138, 89});
  upsample_test({2, 212, 145, 85});
  upsample_test({2, 248, 155, 72});
  upsample_test({2, 217, 153, 83});
  upsample_test({2, 232, 138, 90});
  upsample_test({2, 242, 142, 100});
  upsample_test({2, 218, 131, 75});
  upsample_test({2, 216, 145, 80});
  upsample_test({2, 246, 154, 94});
  upsample_test({2, 211, 143, 89});
  upsample_test({2, 238, 137, 81});
  upsample_test({2, 225, 145, 79});
  upsample_test({2, 242, 147, 94});
  upsample_test({2, 216, 144, 97});
  upsample_test({2, 240, 141, 80});
  upsample_test({2, 227, 131, 76});
  upsample_test({2, 213, 145, 89});
  upsample_test({2, 222, 137, 94});
  upsample_test({2, 218, 155, 88});
  upsample_test({2, 227, 150, 96});
  upsample_test({2, 237, 135, 71});
  upsample_test({2, 232, 138, 86});
  upsample_test({2, 253, 129, 100});
  upsample_test({2, 230, 140, 90});
  upsample_test({2, 215, 136, 100});
  upsample_test({2, 247, 152, 95});
  upsample_test({2, 236, 145, 71});
  upsample_test({2, 225, 149, 82});
  upsample_test({2, 219, 130, 91});
  upsample_test({2, 216, 148, 80});
  upsample_test({2, 225, 151, 80});
  upsample_test({2, 224, 147, 77});
  upsample_test({2, 231, 127, 88});
  upsample_test({2, 222, 130, 92});
  upsample_test({2, 256, 127, 94});
  upsample_test({2, 256, 152, 100});
  upsample_test({2, 239, 134, 88});
  upsample_test({2, 217, 151, 85});
  upsample_test({2, 235, 133, 100});
  upsample_test({2, 211, 133, 86});
  upsample_test({2, 246, 138, 96});
  upsample_test({2, 241, 151, 98});
  upsample_test({2, 229, 130, 98});
  upsample_test({2, 247, 151, 71});
  upsample_test({2, 230, 147, 89});
  upsample_test({2, 239, 141, 95});
  upsample_test({2, 225, 156, 91});
  upsample_test({2, 240, 133, 75});
  upsample_test({2, 212, 137, 80});
  upsample_test({2, 213, 134, 75});
  upsample_test({2, 238, 132, 98});

  for (int i = 0; i < 20; i++) {
    upsample_test({2, 256, 156, 100});
  }

  disable_bucket_refinement();
}
