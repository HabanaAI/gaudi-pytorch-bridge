/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/torch.h>
#include "habana_kernels/fallback_helper.h"

class FallbackTest : public ::testing::Test,
                     public habana_lazy_test::EnvHelper {
  void SetUp() override {
    SetLazyMode();
    SetSeed();
    EnableCpuFallback();
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    RestoreMode();
  }
};

TEST_F(FallbackTest, Simple) {
  auto ones = torch::ones(10, "hpu");
  auto res = ones.digamma();
  res = res.add(ones);

  constexpr float ones_digamma = 0.42278409;
  auto exp = torch::full(10, ones_digamma);
  EXPECT_TRUE(allclose(exp, res.to("cpu")));

  const auto& freq = habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(freq.at("aten::digamma.out"), 1);
}

TEST_F(FallbackTest, Inplace) {
  auto t = torch::rand(10).to("hpu");
  auto res = t.lgamma_();

  EXPECT_EQ(t.storage().data_ptr().get(), res.storage().data_ptr().get());

  const auto& freq = habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(freq.at("aten::lgamma.out"), 1);
}

TEST_F(FallbackTest, inverse) {
  auto a = torch::randn({2, 2});
  auto b = a.inverse();
  auto out = torch::transpose(b, 0, 1);

  auto ha = a.to("hpu");
  auto hb = ha.inverse();
  auto hout = torch::transpose(hb, 0, 1);
  EXPECT_TRUE(allclose(out, hout.to("cpu"), 0.001, 0.001));

  const auto& freq = habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(freq.at("aten::linalg_inv_ex.inverse"), 1);
}

// Test disabled since we do not want to support CPU Fallback for as_strided.
// Enable this test when strided tensors are completely supported on HPU and
// move it to appropriate test file
#if 0
TEST_F(FallbackTest, AsStrided) {
  setenv("PT_HPU_PLACE_ON_CPU", "div_", 1);
  torch::Tensor A = torch::rand({3, 3, 3, 3, 3});
  torch::Tensor hA = A.to(torch::kHPU);
  at::Tensor Out = A.as_strided({2, 2}, {1, 2});
  at::Tensor hOut = hA.as_strided({2, 2}, {1, 2});

  Out.div_(4);
  hOut.div_(4);

  EXPECT_TRUE(allclose(Out, hOut.to("cpu"))) << Out << hOut.to("cpu");
  unsetenv("PT_HPU_PLACE_ON_CPU");
}

TEST_F(FallbackTest, tensorView_Inplace) {
  setenv("PT_HPU_PLACE_ON_CPU", "mul_", 1);
  torch::Tensor tensor = torch::randn({3, 3});
  auto tensor_hpu = tensor.to(torch::kHPU);

  auto out = torch::as_strided(tensor, (2, 2), (1, 2));
  out.mul_(2);

  auto out_hpu = torch::as_strided(tensor_hpu, (2, 2), (1, 2));
  out_hpu.mul_(2);

  auto hOut_cpu = out_hpu.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
  unsetenv("PT_HPU_PLACE_ON_CPU");
}

TEST_F(FallbackTest, tensorView_OutOfPlace) {
  setenv("PT_HPU_PLACE_ON_CPU", "add", 1);
  torch::Tensor tensor = torch::randn({3, 3});
  auto tensor_hpu = tensor.to(torch::kHPU);

  auto out = torch::as_strided(tensor, (2, 2), (1, 2));
  out.add(2);

  auto out_hpu = torch::as_strided(tensor_hpu, (2, 2), (1, 2));
  out_hpu.add(2);

  auto hOut_cpu = out_hpu.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
  unsetenv("PT_HPU_PLACE_ON_CPU");
}

TEST_F(FallbackTest, tensorView_Inplace_2) {
  setenv("PT_HPU_PLACE_ON_CPU", "addcmul_", 1);

  int N = 8;
  int C = 3;
  int H = 24;
  int W = 24;
  auto a = torch::randn({N, C, H, W});
  auto b = torch::randn({N, C, H, W});
  auto c = torch::randn({N, C, H, W});

  auto hpu_a = a.to(torch::kHPU);
  auto hpu_b = b.to(torch::kHPU);
  auto hpu_c = c.to(torch::kHPU);

  a.addcmul_(b, c, 1.0);
  hpu_a.addcmul_(hpu_b, hpu_c, 1.0);

  auto hOut_cpu = hpu_a.cpu();
  EXPECT_EQ(allclose(a, hOut_cpu, 0.001, 0.001), true);

  unsetenv("PT_HPU_PLACE_ON_CPU");
}

TEST_F(FallbackTest, tensorView_Inplace_3) {
  setenv("PT_HPU_PLACE_ON_CPU", "add_", 1);
  torch::Tensor tensor = torch::randn({3, 3});
  auto tensor_hpu = tensor.to(torch::kHPU);

  auto out = torch::as_strided(tensor, (2, 2), (1, 2));
  out.mul_(2);
  tensor.add_(1);
  auto relu = torch::nn::ReLU();
  auto res = relu(tensor);

  auto out_hpu = torch::as_strided(tensor_hpu, (2, 2), (1, 2));
  out_hpu.mul_(2);
  tensor_hpu.add_(1);
  auto res_hpu = relu(tensor_hpu);

  auto hOut_cpu = res_hpu.cpu();
  EXPECT_EQ(allclose(res, hOut_cpu, 0.001, 0.001), true);
  unsetenv("PT_HPU_PLACE_ON_CPU");
}

TEST_F(FallbackTest, tensorView_Inplace_4) {
  setenv("PT_HPU_PLACE_ON_CPU", "bitwise_and_", 1);
  auto a = torch::tensor({-1, -2, 3, 1, 3, 4, 4, 5, 0, 7}, dtype(at::kChar));
  auto b = torch::tensor({1, 0, 3, 0, 1, 0, 4, -4, -8, 0}, dtype(at::kChar));
  auto a_hpu = a.to(torch::kHPU);
  auto b_hpu = b.to(torch::kHPU);

  auto out = torch::as_strided(a, (2, 2), (1, 2));
  a.bitwise_and_(b);
  auto out_hpu = torch::as_strided(a_hpu, (2, 2), (1, 2));
  a_hpu.bitwise_and_(b_hpu);

  auto hOut_cpu = a_hpu.cpu();
  EXPECT_EQ(allclose(a, hOut_cpu, 0.001, 0.001), true);
  unsetenv("PT_HPU_PLACE_ON_CPU");
}

TEST_F(FallbackTest, tensorlistView_Inplace) {
  setenv("PT_HPU_PLACE_ON_CPU", "_foreach_abs_", 1);
  torch::Tensor t1 = torch::randn({3, 3});
  torch::Tensor t2 = torch::randn({8, 8});
  auto t1_hpu = t1.to(torch::kHPU);
  auto t2_hpu = t2.to(torch::kHPU);
  auto view1 = torch::as_strided(t1, (2, 2), (1, 2));
  auto view2 = torch::as_strided(t2, (2, 6), (1, 2));
  at::_foreach_abs_({view1, view2});
  auto view1_hpu = torch::as_strided(t1_hpu, (2, 2), (1, 2));
  auto view2_hpu = torch::as_strided(t2_hpu, (2, 6), (1, 2));
  at::_foreach_abs_({view1_hpu, view2_hpu});

  /*
  foreach ops are not supporting on view tensor
  EXPECT_EQ(allclose(a, hOut_cpu, 0.001, 0.001), true);
  EXPECT_EQ(allclose(a, hOut_cpu, 0.001, 0.001), true);
  */
  unsetenv("PT_HPU_PLACE_ON_CPU");
}
#endif
