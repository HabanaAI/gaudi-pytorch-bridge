/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/torch.h>
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hpu_lazy_tensors.h"

#include "util.h"

class EmbeddingDTypeFallbackTest : public DTypeSupportTest<c10::ScalarType> {};

TEST_P(EmbeddingDTypeFallbackTest, HpuSupportsEmbeddingInt64Indices) {
  auto options = torch::TensorOptions().dtype(GetParam()).device(torch::kHPU);
  torch::Tensor indices = torch::tensor({1, 3}, options);
  torch::Tensor weights = torch::rand({5, 5}).to(torch::kHPU);

  auto result =
      torch::embedding(weights, indices, 0, false, false).to(torch::kCPU);
  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();

  EXPECT_EQ(
      op_fallback_frequency.find("aten::embedding"),
      op_fallback_frequency.end());
};

TEST_P(
    EmbeddingDTypeFallbackTest,
    HpuSupportsEmbeddingDenseBackwardInt64Indices) {
  auto options = torch::TensorOptions().dtype(GetParam()).device(torch::kHPU);
  torch::Tensor indices = torch::tensor({1, 3}, options);
  torch::Tensor grad_out = torch::rand({2, 5}).to(torch::kHPU);

  auto result = torch::embedding_dense_backward(grad_out, indices, 2, 0, false)
                    .to(torch::kCPU);
  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();

  EXPECT_EQ(
      op_fallback_frequency.find("aten::embedding_dense_backward"),
      op_fallback_frequency.end());
}

INSTANTIATE_TEST_SUITE_P(
    TypeSupportTests,
    EmbeddingDTypeFallbackTest,
    testing::Values(torch::kInt32, torch::kInt64));
