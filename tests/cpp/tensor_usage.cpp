/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/logging.h"

#include "backend/synapse_helpers/env_flags.h"

class TensorUsage : public ::testing::Test {
 public:
  TensorUsage() : V{1.0, 2.0, 3.0, 4.0, 5.0} {}

 protected:
  void SetUp() override {}
  std::vector<float> V{};
};

TEST_F(TensorUsage, Clone) {
  auto A_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  int64_t A_numel{static_cast<int64_t>(V.size())};
  torch::Tensor A = torch::from_blob(V.data(), {A_numel}, A_options);
  torch::Tensor A_clone = A.clone().detach();

  PT_TEST_DEBUG("PTI_DBG :: data addr : ", V.data());
  PRINT_TENSOR_WITH_DATA(A);
  PRINT_TENSOR_WITH_DATA(A_clone);
  EXPECT_EQ(allclose(A, A_clone, 0.01, 0.01), true);

  for (size_t i{0}; i < V.size(); i++) {
    V[i] *= 1111;
  }

  PRINT_TENSOR_WITH_DATA(A);
  PRINT_TENSOR_WITH_DATA(A_clone);

  EXPECT_NE(allclose(A, A_clone, 0.01, 0.01), true);
}
