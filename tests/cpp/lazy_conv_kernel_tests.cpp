/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyConvKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyConvKernelTest, ConvReluTest) {
  auto input_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 3, 3}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);

  auto weight_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({3, 3, 3, 1}); // hwck

  torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);
  torch::Tensor outConv =
      torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::Tensor outhpu = torch::relu(outConv);
  torch::Tensor out = outhpu.to(torch::kCPU);

  torch::Tensor outConv1 = torch::conv2d(
      input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::Tensor outcpu = torch::relu(outConv1);

  EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
}

TEST_F(LazyConvKernelTest, ConvReluSynapsePermutationTest) {
  for (size_t i = 0; i < 2; ++i) {
    auto input_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 3, 3}); // nchw
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);

    auto weight_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({3, 3, 3, 1}); // hwck

    torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);

    torch::Tensor outConv =
        torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outhpu = torch::relu(outConv);
    torch::Tensor out = outhpu.to(torch::kCPU);

    torch::Tensor outConv1 = torch::conv2d(
        input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outcpu = torch::relu(outConv1);

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE)) {
      HbLazyTensor weight_hb_tensor = SyncAndGetHbLazyTensor(outhpu);
      auto hb_wight_data = weight_hb_tensor.EvaluateTensorData();
      auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(hb_wight_data);
      auto perm = hb_weight_impl->GetMemoryPermutation();
      std::vector<uint8_t> expected_perm = {2, 0, 1, 3};
      EXPECT_EQ(perm, expected_perm);
      EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
    }
    auto weight_tensor1 =
        torch::arange(9, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({3, 3, 1, 1}); // hwck

    torch::Tensor tHabanaW1 = weight_tensor1.to(torch::kHPU);
    torch::Tensor outConv2 =
        torch::conv2d(outhpu, tHabanaW1, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    torch::Tensor outConv2Cpu =
        torch::conv2d(out, weight_tensor1, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    torch::Tensor out1 = outConv2.to(torch::kCPU);
    EXPECT_EQ(allclose(out1, outConv2Cpu, 0.01, 0.01), true);
  }
}

TEST_F(LazyConvKernelTest, ConvReluSynapsePermutationTest2) {
  auto weight_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({3, 3, 3, 1}); // hwck

  torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);

  auto weight_tensor1 =
      torch::arange(9, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({3, 3, 1, 1}); // hwck

  torch::Tensor tHabanaW1 = weight_tensor1.to(torch::kHPU);
  for (size_t i = 0; i < 2; ++i) {
    auto input_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 3, 3}); // nchw
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);

    torch::Tensor outConv =
        torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    torch::Tensor tHabanaWcpu = tHabanaW.to(torch::kCPU);

    EXPECT_EQ(allclose(tHabanaWcpu, weight_tensor, 0.01, 0.01), true);

    torch::Tensor outhpu = torch::relu(outConv);
    torch::Tensor out = outhpu.to(torch::kCPU);

    torch::Tensor outConv1 = torch::conv2d(
        input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outcpu = torch::relu(outConv1);

    EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);

    torch::Tensor outConv2 =
        torch::conv2d(outhpu, tHabanaW1, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    torch::Tensor outConv2Cpu =
        torch::conv2d(out, weight_tensor1, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE)) {
      HbLazyTensor weight_hb_tensor = SyncAndGetHbLazyTensor(tHabanaW);
      auto hb_wight_data = weight_hb_tensor.EvaluateTensorData();
      auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(hb_wight_data);
      auto perm = hb_weight_impl->GetMemoryPermutation();
      std::vector<uint8_t> expected_perm = {3, 2, 0, 1};
      EXPECT_EQ(perm, expected_perm);
    }

    HbLazyTensor::StepMarker({});

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE)) {
      HbLazyTensor weight_hb_tensor = SyncAndGetHbLazyTensor(outConv2);
      auto hb_wight_data = weight_hb_tensor.EvaluateTensorData();
      auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(hb_wight_data);
      auto perm = hb_weight_impl->GetMemoryPermutation();
      std::vector<uint8_t> expected_perm = {2, 0, 1, 3};
    }
    torch::Tensor out1 = outConv2.to(torch::kCPU);

    EXPECT_EQ(allclose(out1, outConv2Cpu, 0.01, 0.01), true);

    torch::Tensor tHabanaW1cpu = tHabanaW1.to(torch::kCPU);

    EXPECT_EQ(allclose(tHabanaW1cpu, weight_tensor1, 0.01, 0.01), true);

    tHabanaW1.add_(1);
    weight_tensor1.add_(1);

    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE) &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE)) {
      HbLazyTensor weight_hb_tensor = SyncAndGetHbLazyTensor(tHabanaW1);
      auto hb_wight_data = weight_hb_tensor.EvaluateTensorData();
      auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(hb_wight_data);
      auto perm = hb_weight_impl->GetMemoryPermutation();
      std::vector<uint8_t> expected_perm = {3, 2, 0, 1};
      EXPECT_EQ(perm, expected_perm);
    }

    tHabanaW1cpu = tHabanaW1.to(torch::kCPU);

    EXPECT_EQ(allclose(tHabanaW1cpu, weight_tensor1, 0.01, 0.01), true);

    tHabanaW1 = weight_tensor1.to(torch::kHPU);
    EXPECT_EQ(allclose(tHabanaW1, weight_tensor1, 0.01, 0.01), true);
    auto weight_hb_tensor = SyncAndGetHbLazyTensor(tHabanaW1);
    auto hb_wight_data = weight_hb_tensor.EvaluateTensorData();
    auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(hb_wight_data);
    auto perm = hb_weight_impl->GetMemoryPermutation();
    std::vector<uint8_t> expected_perm = {};
    // verify that Memory permutation was removed
    EXPECT_EQ(perm, expected_perm);
  }
}

TEST_F(LazyConvKernelTest, MaxPool2DTest) {
  auto input_tensor = torch::randn({20, 16, 50, 32});

  torch::Tensor outHabana = torch::max_pool2d(input_tensor, 2, 1);
  auto out = outHabana.to(torch::kCPU);

  torch::Tensor outcpu = torch::max_pool2d(input_tensor, 2, 1);

  EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
}

class LazyConvKernelGraphTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    ForceMode(1);
    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }
};

// Also validates OutputShapeInf for Conv Bwd
TEST_F(LazyConvKernelGraphTest, ConvolutionBackward) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  auto grad_output = torch::randn({2, 6, 2, 3}, torch::requires_grad(false));
  auto input = torch::randn({2, 5, 3, 4}, torch::requires_grad(false));
  auto weight = torch::randn({6, 5, 2, 2}, torch::requires_grad(false));

  auto h_grad_output = grad_output.to(torch::kHPU);
  auto hinput = input.to(torch::kHPU);
  auto hweight = weight.to(torch::kHPU);
  torch::Tensor out1, out2, out3;
  std::tie(out1, out2, out3) = convolution_backward_overrideable(
      h_grad_output,
      hinput,
      hweight,
      {1, 1},
      {0, 0},
      {1, 1},
      false,
      {0, 0},
      1,
      {1, 1, 1});

  std::vector<HbLazyTensor> tensors = {
      SyncAndGetHbLazyTensor(out1),
      SyncAndGetHbLazyTensor(out2),
      SyncAndGetHbLazyTensor(out3)};

  std::vector<ir::NodePtr> a{tensors[0].CurrentIrValue().mp_node};
  std::vector<int> indices1{0, 1, 2};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices1);

  auto hl_grad_output = SyncAndGetHbLazyTensor(h_grad_output);
  auto hl_input = SyncAndGetHbLazyTensor(hinput);
  auto hl_weight = SyncAndGetHbLazyTensor(hweight);
  std::vector<at::Tensor> input_list{
      hl_grad_output.EvaluateTensorData(),
      hl_input.EvaluateTensorData(),
      hl_weight.EvaluateTensorData()};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check("= prim::Constant[value=[1, 1]]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("= prim::Constant[value=[0, 0]]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("= prim::Constant[value=0]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("= prim::Constant[value=1]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("= prim::Constant[value=[True, True, True]]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check_count("= aten::convolution_backward_overrideable", 1)
      ->run(*hlexec->get_graph());
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates OutputShapeInf for Conv2d
TEST_F(LazyConvKernelTest, ConvExecTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  auto in = torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({5, 4, 3, 3}, torch::dtype(torch::kFloat)); // kchw
  auto exp = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);

  torch::Tensor result =
      torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates OutputShapeInf for ConvTranspose2d
TEST_F(LazyConvKernelTest, ConvTranspose2dTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  auto in = torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({4, 5, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
  auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
  auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);

  torch::Tensor result = torch::conv_transpose2d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates OutputShapeInf for ConvTranspose2d
TEST_F(LazyConvKernelTest, ConvTranspose2dG2Test) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  auto in = torch::randn({1, 16, 12, 12}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({16, 8, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
  auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
  auto exp = torch::conv_transpose2d(in, wt, {}, 2, 0, 0, 2, 1);

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);
  torch::Tensor result = torch::conv_transpose2d(h_in, h_wt, {}, 2, 0, 0, 2, 1);
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyConvKernelTest, Conv2dG2Test) {
  auto in = torch::randn({1, 16, 12, 12}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({16, 8, 3, 3}, torch::dtype(torch::kFloat)); // kchw
  auto exp =
      torch::relu(torch::conv2d(in, wt, {}, {2}, at::IntArrayRef{0}, {1}, 2));

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);
  torch::Tensor outConv1 =
      torch::conv2d(h_in, h_wt, {}, {2}, at::IntArrayRef{0}, {1}, 2);
  torch::Tensor outcpu = torch::relu(outConv1);
  Tensor out = outcpu.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

// Also validates OutputShapeInf for ConvTranspose3d
TEST_F(LazyConvKernelTest, ConvTranspose3dTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  auto in =
      torch::randn({32, 3, 4, 14, 14}, torch::dtype(torch::kFloat)); // ncdhw
  auto wt = torch::randn({3, 5, 3, 3, 3}, torch::dtype(torch::kFloat)); // cktrs
  auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
  auto exp = torch::conv_transpose3d(in, wt, {}, 1, 0, 0, 1, 1);

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);

  torch::Tensor result = torch::conv_transpose3d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates OutputShapeInf for ConvTranspose3d
TEST_F(LazyConvKernelTest, ConvTranspose3dG2Test) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  auto in =
      torch::randn({32, 8, 4, 14, 14}, torch::dtype(torch::kFloat)); // ncdhw
  auto wt = torch::randn({8, 4, 3, 3, 3}, torch::dtype(torch::kFloat)); // cktrs
  auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
  auto exp = torch::conv_transpose3d(in, wt, {}, 2, 0, 0, 2, 1);

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);

  torch::Tensor result = torch::conv_transpose3d(h_in, h_wt, {}, 2, 0, 0, 2, 1);
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyConvKernelTest, Conv3dTest) {
  auto in =
      torch::randn({64, 5, 4, 28, 28}, torch::dtype(torch::kFloat)); // ncdhw
  auto wt = torch::randn({3, 5, 3, 3, 3}, torch::dtype(torch::kFloat)); // cktrs
  auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
  auto exp =
      torch::relu(torch::conv3d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1));

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);

  torch::Tensor result = torch::relu(
      torch::conv3d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1));
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

TEST_F(LazyConvKernelTest, Conv3dG2Test) {
  auto in =
      torch::randn({64, 16, 4, 28, 28}, torch::dtype(torch::kFloat)); // ncdhw
  auto wt =
      torch::randn({16, 8, 3, 3, 3}, torch::dtype(torch::kFloat)); // cktrs
  auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
  auto exp =
      torch::relu(torch::conv3d(in, wt, {}, {2}, at::IntArrayRef{0}, {1}, 2));

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);

  torch::Tensor result = torch::relu(
      torch::conv3d(h_in, h_wt, {}, {2}, at::IntArrayRef{0}, {1}, 2));
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

// Also validates OutputShapeInf for Conv2d
TEST_F(LazyConvKernelTest, ConvInferenceTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE))
    SET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE, true, 1);

  auto in = torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({5, 4, 3, 3}, torch::dtype(torch::kFloat)); // kchw
  auto exp = torch::conv2d(in, wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  auto h_in = in.to(torch::kHPU);
  auto h_wt = wt.to(torch::kHPU);

  torch::Tensor result =
      torch::conv2d(h_in, h_wt, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE);
}
