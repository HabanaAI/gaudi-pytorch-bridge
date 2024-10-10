/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include <ATen/ExpandUtils.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>
#include "backend/synapse_helpers/env_flags.h"
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"

using namespace habana_lazy;

TEST(NMSTest, NmsSmall) {
  torch::manual_seed(0);
  // Generate random scores for each box
  auto num_boxes = 10;
  torch::Tensor scores = torch::rand({num_boxes});
  torch::Tensor hscores = scores.to(torch::kHPU);

  // Generate boxes of random sizes
  torch::Tensor boxes = torch::rand({num_boxes, 4}) * 256;
  // ensure x2 > x1 and y2 > y1
  auto tlist = boxes.split(2, 1);
  tlist[1] = tlist[1] + tlist[0];
  auto new_boxes = torch::cat({tlist[0], tlist[1]}, 1);
  torch::Tensor hboxes = new_boxes.to(torch::kHPU);

  auto nms_boxid = torchvision_nms_hpu_wrap(hboxes, hscores, 0.2);
  auto ref = torch::tensor({7, 1, 5, 0, 6, 8, 4}).to(torch::kLong);
  bool equal = ref.allclose(nms_boxid.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST(NMSTest, BatchedNms) {
  torch::manual_seed(0);
  // Generate random scores for each box
  auto num_boxes = 10;
  torch::Tensor scores = torch::rand({num_boxes});
  torch::Tensor hscores = scores.to(torch::kHPU);

  // Generate boxes of random sizes
  torch::Tensor boxes = torch::rand({num_boxes, 4}) * 256;
  // ensure x2 > x1 and y2 > y1
  auto tlist = boxes.split(2, 1);
  tlist[1] = tlist[1] + tlist[0];
  auto new_boxes = torch::cat({tlist[0], tlist[1]}, 1);
  torch::Tensor hboxes = new_boxes.to(torch::kHPU);

  torch::Tensor classes = torch::randint(0, 1, {10}, torch::kInt32);
  torch::Tensor hclasses = classes.to(torch::kHPU);

  auto nms_boxid = batched_nms_hpu_lazy(hboxes, hscores, hclasses, 0.2);
  auto ref = torch::tensor({7, 1, 5, 0, 6, 8, 4}).to(torch::kLong);
  bool equal = ref.allclose(nms_boxid.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}
