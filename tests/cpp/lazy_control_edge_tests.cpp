/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "habana_lazy_test_infra.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

using namespace habana_lazy;

class LazyControlEdgeTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyControlEdgeTest, AsStridedWithinOps) {
  torch::Tensor c0 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor c1 = torch::as_strided(c0, {20, 5}, {5, 1});

  torch::Tensor c2 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor c3 = c2.abs();
  torch::Tensor c4 = torch::as_strided(c3, {20, 5}, {5, 1});
  c4.copy_(c1);
  torch::Tensor c5 = c3.relu();

  torch::Tensor h0 = c0.to(torch::kHABANA);
  torch::Tensor h1 = torch::as_strided(h0, {20, 5}, {5, 1});
  torch::Tensor h2 = c2.to(torch::kHABANA);
  torch::Tensor h3 = h2.abs();
  torch::Tensor h4 = torch::as_strided(h3, {20, 5}, {5, 1});
  h4.copy_(h1);
  torch::Tensor h5 = h3.relu();

  torch::Tensor h5_c = h5.to(torch::kCPU);

  EXPECT_TRUE(allclose(c5, h5_c));
}
