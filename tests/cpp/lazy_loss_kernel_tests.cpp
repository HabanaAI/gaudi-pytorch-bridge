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
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using namespace habana_lazy;
using namespace at;
class LazyLossKernelTest : public habana_lazy_test::LazyTest {};

class LazyLossKernelWithParamsTest
    : public habana_lazy_test::LazyTest,
      public ::testing::WithParamInterface<
          std::tuple<bool, bool, at::Reduction::Reduction>> {
 protected:
  void TestBCELossLogits(
      bool testWeight,
      bool testPosWeight,
      at::Reduction::Reduction reductionType) {
    auto input = torch::randn({5, 2, 4, 3});
    auto target = torch::randn({5, 2, 4, 3});
    auto grad_output = torch::randn({1});
    // weight (Tensor, optional) – a manual rescaling weight if provided it’s
    // repeated to match input tensor shape
    c10::optional<Tensor> weight =
        testWeight ? torch::randn({3}) : c10::optional<Tensor>();
    // pos_weight (Tensor, optional) – a weight of positive examples. Must be a
    // vector with length equal to the number of classes.
    c10::optional<Tensor> pos_weight =
        testPosWeight ? torch::rand({3}) : c10::optional<Tensor>();

    torch::Tensor hinput = input.to(torch::kHPU);
    torch::Tensor htarget = target.to(torch::kHPU);
    torch::Tensor hgrad_out = grad_output.to(torch::kHPU);
    c10::optional<Tensor> hweight =
        testWeight ? weight.value().to(torch::kHPU) : c10::optional<Tensor>();
    c10::optional<Tensor> hpos_weight = testPosWeight
        ? pos_weight.value().to(torch::kHPU)
        : c10::optional<Tensor>();

    auto houtput = torch::binary_cross_entropy_with_logits(
        hinput, htarget, hweight, hpos_weight, reductionType);

    auto houtfwd = houtput.to(torch::kCPU);

    // reference output
    auto expfwd = torch::binary_cross_entropy_with_logits(
        input, target, weight, pos_weight, reductionType);

    EXPECT_EQ(allclose(houtfwd, expfwd, 0.001, 0.001), true);
  }
};

TEST_F(LazyLossKernelTest, MseLossTest) {
  torch::Tensor input = torch::randn({3, 5});
  torch::Tensor target = torch::randn({3, 5});
  torch::Tensor grad_input = torch::randn({3, 5});

  auto hinput = input.to(torch::kHPU);
  auto htarget = target.to(torch::kHPU);
  auto hgrad_input = grad_input.to(torch::kHPU);
  torch::Tensor hout1 = torch::mse_loss(hinput, htarget, at::Reduction::None);
  torch::Tensor hout2 = torch::mse_loss_backward(
      hgrad_input, hinput, htarget, at::Reduction::None);

  std::vector<HbLazyTensor> tensors = {
      SyncAndGetHbLazyTensor(hout1), SyncAndGetHbLazyTensor(hout2)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto out1 = hout1.to(torch::kCPU);
  auto out2 = hout2.to(torch::kCPU);

  auto exp1 = mse_loss(input, target, at::Reduction::None);
  auto exp2 = mse_loss_backward(grad_input, input, target, at::Reduction::None);

  EXPECT_EQ(allclose(out1, exp1), true);
  EXPECT_EQ(allclose(out2, exp2), true);
}

TEST_F(LazyLossKernelTest, KLDivLossTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  auto kl_div_test = [](Tensor& input,
                        Tensor& target,
                        Tensor& grad_out,
                        int64_t reduction,
                        bool log_target,
                        bool is_zero) {
    if (!is_zero) {
      input = torch::_softmax(input, 0, false);
      target = torch::_softmax(target, 0, false);
    }

    auto hinput = input.to(torch::kHPU);
    auto htarget = target.to(torch::kHPU);
    auto hgrad_out = grad_out.to(torch::kHPU);

    torch::Tensor hout1 = torch::kl_div(hinput, htarget, reduction, log_target);

    auto out1 = hout1.to(torch::kCPU);

    auto exp1 = kl_div(input, target, reduction, log_target);

    EXPECT_EQ(allclose(out1, exp1), true);
  };

  // 1D test case.
  torch::Tensor input = torch::randn({9});
  torch::Tensor target = torch::randn({9});
  torch::Tensor grad_out = torch::randn({1});
  torch::Tensor grad_out_none = torch::randn({9});
  auto reduction = at::Reduction::Mean;
  kl_div_test(input, target, grad_out, reduction, true, false);
  kl_div_test(input, target, grad_out, reduction, false, false);
  reduction = at::Reduction::Sum;
  kl_div_test(input, target, grad_out, reduction, true, false);
  kl_div_test(input, target, grad_out, reduction, false, false);
  reduction = at::Reduction::None;
  kl_div_test(input, target, grad_out_none, reduction, true, false);
  kl_div_test(input, target, grad_out_none, reduction, false, false);

  ////3D test case.
  input = torch::randn({4, 2, 2});
  target = torch::randn({4, 2, 2});
  grad_out_none = torch::randn({4, 2, 2});
  grad_out = torch::randn({1});
  reduction = at::Reduction::Mean;
  kl_div_test(input, target, grad_out, reduction, true, false);
  kl_div_test(input, target, grad_out, reduction, false, false);
  reduction = at::Reduction::Sum;
  kl_div_test(input, target, grad_out, reduction, true, false);
  kl_div_test(input, target, grad_out, reduction, false, false);
  reduction = at::Reduction::None;
  kl_div_test(input, target, grad_out_none, reduction, true, false);
  kl_div_test(input, target, grad_out_none, reduction, false, false);

  // 0 values test case.
  input = torch::tensor(
      {0.0, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.25, 0.0},
      torch::dtype(torch::kFloat));
  target = torch::tensor(
      {0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0},
      torch::dtype(torch::kFloat));
  grad_out = torch::randn({1}, torch::dtype(torch::kFloat));
  grad_out_none = torch::randn({9}, torch::dtype(torch::kFloat));
  reduction = at::Reduction::Mean;
  kl_div_test(input, target, grad_out, reduction, true, false);
  kl_div_test(input, target, grad_out, reduction, false, true);
  reduction = at::Reduction::Sum;
  kl_div_test(input, target, grad_out, reduction, true, true);
  kl_div_test(input, target, grad_out, reduction, false, true);
  reduction = at::Reduction::None;
  kl_div_test(input, target, grad_out_none, reduction, true, true);
  kl_div_test(input, target, grad_out_none, reduction, false, true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyLossKernelTest, NllLossFwdTest) {
  torch::Tensor input = torch::randn({10, 4}, torch::requires_grad(true));
  torch::Tensor hinput = input.to(torch::kHPU);

  auto target = torch::randint(
      0,
      3,
      {
          10,
      },
      torch::kLong);
  torch::Tensor htarget = target.to(torch::kHPU);

  torch::nn::NLLLoss loss;
  auto output_cpu = loss->forward(input, target);
  auto output = loss->forward(hinput, htarget);

  Tensor output_hpu = output.to(torch::kCPU);
  EXPECT_EQ(allclose(output_cpu, output_hpu), true);
}

TEST_F(LazyLossKernelTest, NllLoss2dNHWCFwdTest) {
  torch::Tensor input =
      torch::randn({3, 5, 24, 18}, torch::requires_grad(true)); // nchw
  torch::Tensor hinput = input.to(
      torch::kHPU,
      c10::ScalarType::Float,
      false,
      false,
      c10::MemoryFormat::ChannelsLast); // nhwc
  torch::Tensor cinput = hinput.to(torch::kCPU); // nhwc

  auto target = torch::randint(0, 4, {3, 24, 18}, torch::kLong);
  torch::Tensor htarget = target.to(torch::kHPU);

  torch::nn::NLLLoss loss;
  auto output_cpu = loss->forward(cinput, target);
  auto output = loss->forward(hinput, htarget);

  auto output_hpu = output.to(torch::kCPU);
  EXPECT_EQ(allclose(output_cpu, output_hpu, 0.001, 0.001), true);
}

TEST_F(LazyLossKernelTest, NllLoss2dFwdTest) {
  torch::Tensor input =
      torch::randn({6, 4, 18, 24}, torch::requires_grad(true));
  torch::Tensor hinput = input.to(torch::kHPU);

  auto target = torch::randint(0, 3, {6, 18, 24}, torch::kLong);
  torch::Tensor htarget = target.to(torch::kHPU);

  torch::nn::NLLLoss loss;
  auto output_cpu = loss->forward(input, target);
  auto output = loss->forward(hinput, htarget);

  auto output_hpu = output.to(torch::kCPU);
  EXPECT_EQ(allclose(output_cpu, output_hpu, 0.001, 0.001), true);
}

TEST_F(LazyLossKernelTest, NllLossBwdTest) {
  torch::Tensor input = torch::randn({10, 4}, torch::requires_grad(true));
  torch::Tensor hinput = input.to(torch::kHPU);

  auto target = torch::randint(
      0,
      3,
      {
          10,
      },
      torch::kLong);
  torch::Tensor htarget = target.to(torch::kHPU);

  auto grad_out = torch::tensor({1}, torch::kFloat);
  torch::Tensor hgrad_out = grad_out.to(torch::kHPU);

  // HPU kernel does not use this tensor, but we need to create it because
  // "nll_loss_backward" does not compile without this argument. Note that dim &
  // values in this tensor may need to be changed for other "reduction" modes.
  constexpr int sumweights_int = 10;
  auto sum_weights = torch::tensor({sumweights_int}, torch::kFloat);
  torch::Tensor hsum_weights = sum_weights.to(torch::kHPU);

  auto grad_in_cpu = torch::nll_loss_backward(
      grad_out, input, target, {}, 1, -100, sum_weights);
  auto grad_in = torch::nll_loss_backward(
      hgrad_out, hinput, htarget, {}, 1, -100, hsum_weights);

  Tensor grad_in_hpu = grad_in.to(torch::kCPU);
  EXPECT_EQ(allclose(grad_in_cpu, grad_in_hpu), true);
}

TEST_F(LazyLossKernelTest, NllLoss2dBwdTest) {
  torch::Tensor input =
      torch::randn({6, 4, 18, 24}, torch::requires_grad(true));
  torch::Tensor hinput = input.to(torch::kHPU);

  auto target = torch::randint(0, 3, {6, 18, 24}, torch::kLong);
  torch::Tensor htarget = target.to(torch::kHPU);

  auto grad_out = torch::tensor({1}, torch::kFloat);
  torch::Tensor hgrad_out = grad_out.to(torch::kHPU);

  // HPU kernel does not use this tensor, but we need to create it because
  // "nll_loss_backward" does not compile without this argument. Note that dim &
  // values in this tensor may need to be changed for other "reduction" modes.
  // (N,C,H,W) -> (N,H,W,C)
  // 6*18*24 = 2592
  constexpr int sumweights_int = 6 * 18 * 24;
  auto sum_weights = torch::tensor({sumweights_int}, torch::kFloat);
  torch::Tensor hsum_weights = sum_weights.to(torch::kHPU);

  auto grad_in_cpu = torch::nll_loss2d_backward(
      grad_out, input, target, {}, 1, -100, sum_weights);
  auto grad_in = torch::nll_loss2d_backward(
      hgrad_out, hinput, htarget, {}, 1, -100, hsum_weights);

  Tensor grad_in_hpu = grad_in.to(torch::kCPU);
  EXPECT_EQ(allclose(grad_in_cpu, grad_in_hpu, 0.001, 0.001), true);
}

TEST_F(LazyLossKernelTest, NllLoss2dBwdTest_DynamicShape) {
  torch::Tensor input =
      torch::randn({12, 4, 18, 16}, torch::requires_grad(true));
  torch::Tensor hinput = input.to(torch::kHPU);

  auto target = torch::randint(
      0,
      3,
      {
          12,
          18,
          16,
      },
      torch::kLong);
  torch::Tensor htarget = target.to(torch::kHPU);

  auto grad_out = torch::tensor({1}, torch::kFloat);
  torch::Tensor hgrad_out = grad_out.to(torch::kHPU);

  // HPU kernel does not use this tensor, but we need to create it because
  // "nll_loss_backward" does not compile without this argument. Note that dim
  // & values in this tensor may need to be changed for other "reduction"
  // modes. (N,C,H,W) -> (N,H,W,C) 12*18*16 = 3456
  constexpr int sumweights_int = 12 * 18 * 16;
  auto sum_weights = torch::tensor({sumweights_int}, torch::kFloat);
  torch::Tensor hsum_weights = sum_weights.to(torch::kHPU);

  auto grad_in_cpu = torch::nll_loss2d_backward(
      grad_out, input, target, {}, 1, -100, sum_weights);
  auto grad_in = torch::nll_loss2d_backward(
      hgrad_out, hinput, htarget, {}, 1, -100, hsum_weights);

  Tensor grad_in_hpu = grad_in.to(torch::kCPU);
  EXPECT_EQ(allclose(grad_in_cpu, grad_in_hpu, 0.001, 0.001), true);
}

TEST_F(LazyLossKernelTest, BCELossTest) {
  auto input = torch::randn({6, 1});
  auto target = torch::rand({6, 1}); // Nx1
  auto grad_output = torch::randn({1});

  torch::Tensor hinput = input.to(torch::kHPU);
  torch::Tensor htarget = target.to(torch::kHPU);
  torch::Tensor hgrad_out = grad_output.to(torch::kHPU);

  auto hsigmout = torch::sigmoid(hinput);
  auto houtput =
      torch::binary_cross_entropy(hsigmout, htarget, {}, at::Reduction::Mean);
  auto hboutput = torch::binary_cross_entropy_backward(
      hgrad_out, hsigmout, htarget, {}, at::Reduction::Mean);

  auto houtfwd = houtput.to(torch::kCPU);
  auto houtbwd = hboutput.to(torch::kCPU);

  // reference output
  auto expfwd = torch::binary_cross_entropy(
      torch::sigmoid(input), target, {}, at::Reduction::Mean);
  auto expbwd = torch::binary_cross_entropy_backward(
      grad_output, torch::sigmoid(input), target, {}, at::Reduction::Mean);

  EXPECT_EQ(allclose(houtfwd, expfwd), true);
  EXPECT_EQ(allclose(houtbwd, expbwd), true);
}

// Also validates InferOutputMeta for BCELogitsFwd
TEST_F(LazyLossKernelTest, BCELogitsFwdLossTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  auto input = torch::randn({5, 2, 4, 3});
  auto target = torch::randn({5, 2, 4, 3});
  // weight (Tensor, optional) – a manual rescaling weight if provided it’s
  // repeated to match input tensor shape
  c10::optional<Tensor> weight = torch::randn({3});
  // pos_weight (Tensor, optional) – a weight of positive examples. Must be a
  // vector with length equal to the number of classes.
  c10::optional<Tensor> pos_weight = torch::rand({3});

  torch::Tensor hinput = input.to(torch::kHPU);
  torch::Tensor htarget = target.to(torch::kHPU);
  c10::optional<Tensor> hweight = weight.value().to(torch::kHPU);
  c10::optional<Tensor> hpos_weight = pos_weight.value().to(torch::kHPU);

  auto houtput = torch::binary_cross_entropy_with_logits(
      hinput, htarget, hweight, hpos_weight, at::Reduction::Mean);

  auto houtfwd = houtput.to(torch::kCPU);

  // reference output
  auto expfwd = torch::binary_cross_entropy_with_logits(
      input, target, weight, pos_weight, at::Reduction::Mean);

  EXPECT_EQ(allclose(houtfwd, expfwd, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_P(LazyLossKernelWithParamsTest, BCELogitsLossTest) {
  bool testWeight = std::get<0>(GetParam());
  bool testPosWeight = std::get<1>(GetParam());
  at::Reduction::Reduction reductionType = std::get<2>(GetParam());
  std::cout << "testWeight: " << testWeight
            << " testPosWeight: " << testPosWeight
            << " reductionType: " << reductionType << std::endl;
  // TODO: Remove when this is resolved:
  // https://jira.habana-labs.com/browse/SW-67715
  setenv("PT_HPU_LAZY_CACHE_DISABLE", "true", 1);
  TestBCELossLogits(testWeight, testPosWeight, reductionType);
  // TODO: Remove when this is resolved:
  // https://jira.habana-labs.com/browse/SW-67715
  unsetenv("PT_HPU_LAZY_CACHE_DISABLE");
}

const auto testWeights = testing::Values(false, true);

const auto testPosWeights = testing::Values(false, true);

const auto reductionTypesToTest =
    testing::Values(at::Reduction::Sum, at::Reduction::Mean);

INSTANTIATE_TEST_CASE_P(
    BCELogitsLossTest,
    LazyLossKernelWithParamsTest,
    ::testing::Combine(testWeights, testPosWeights, reductionTypesToTest));
