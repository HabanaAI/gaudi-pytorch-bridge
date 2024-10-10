/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy/lazy_executor.h"

using namespace habana_lazy;
using namespace at;

class LazyTensorAPITest : public habana_lazy_test::LazyTest {};

TEST_F(LazyTensorAPITest, NumelDimSizeTest) {
  torch::Tensor A = torch::tensor(
      {{{2, 4, 1, 3, 3}, {0, 9, 8, 7, 6}, {7, 7, 7, 8, 8}},
       {{2, 4, 1, 3, 3}, {9, 9, 1, 3, -2}, {8, 3, 2, 1, 0}}});
  torch::Tensor B = torch::tensor(
      {{{2, 6, 1, 1, 0}, {9, 2, 5, 6, -5}, {8, 5, 2, 1, 7}},
       {{1, 5, 1, 5, 1}, {1, 4, 1, 3, -2}, {1, 6, 8, 9, 10}}});
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out = torch::mul(hA, hB);

  ASSERT_TRUE(out.numel() == 30);
  ASSERT_TRUE(out.dim() == 3);
  ASSERT_TRUE(out.size(0) == 2);
  ASSERT_TRUE(out.size(1) == 3);
  ASSERT_TRUE(out.size(2) == 5);
}

TEST_F(LazyTensorAPITest, EmptyStorage) {
  auto a = torch::empty(4, "hpu");
  habana_lazy::HbLazyTensor::StepMarker();
  ASSERT_TRUE(SyncAndGetHbLazyTensor(a).CurrentTensorData() != nullopt);
}

TEST_F(LazyTensorAPITest, DataPtr) {
  auto dummy = torch::ones(1).to("hpu");
  auto p = habana_lazy::HbLazyTensor::lazyTensorDataPtr(dummy);
  PT_TEST_DEBUG("tensor data ptr = ", p, "\n");
  auto dummy2 = torch::randn({2, 300, 5, 6}).to("hpu");
  auto p2 = habana_lazy::HbLazyTensor::lazyTensorDataPtr(dummy2);
  PT_TEST_DEBUG("tensor 2 data ptr = ", p2, "\n");
  ASSERT_TRUE(p != p2);
  auto dummy3 = torch::randn({2, 3, 5, 6}).to("hpu");
  auto p3 = habana_lazy::HbLazyTensor::lazyTensorDataPtr(dummy3);
  PT_TEST_DEBUG("tensor 3 data ptr = ", p3, "\n");
  ASSERT_TRUE(p != p3 && p2 != p3);
  auto dummy4 = torch::randn({2}).to("hpu");
  auto p4 = habana_lazy::HbLazyTensor::lazyTensorDataPtr(dummy4);
  PT_TEST_DEBUG("tensor 4 data ptr = ", p4, "\n");
  ASSERT_TRUE(p != p4 && p2 != p4 && p3 != p4);
  auto p_again = habana_lazy::HbLazyTensor::lazyTensorDataPtr(dummy);
  PT_TEST_DEBUG("tensor data ptr second call = ", p_again, "\n");
  ASSERT_TRUE(p == p_again);
}

TEST_F(LazyTensorAPITest, ShapeTensorTest) {
  LazyExecutionMode exec_mode{get_habana_lazy_executor().getExecutionMode()};
  get_habana_lazy_executor().setExecutionMode(LazyExecutionMode::kLOWERING);
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  auto syn_graph =
      habana_helpers::create_graph(device.id(), "Test_graph", false);
  torch::Tensor input = torch::randn({10, 20}).to(torch::kHPU);
  auto syn_shape_input = habana_helpers::create_shape_tensor(
      input, syn_graph, false, SHAPE_TENSOR);
  ASSERT_TRUE(syn_shape_input.is_persistent());
  get_habana_lazy_executor().setExecutionMode(exec_mode);
}
