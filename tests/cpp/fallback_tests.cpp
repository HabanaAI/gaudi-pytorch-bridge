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
}

TEST_F(FallbackTest, UnsupportedOpHalf) {
  auto in =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kHalf));
  auto res = torch::tril(in);

  auto exp = torch::tensor({{1, 0}, {3, 4}}, dtype(at::kHalf));
  EXPECT_TRUE(allclose(exp, res.to("cpu")));
}

TEST_F(FallbackTest, Inplace) {
  auto t = torch::rand(10).to("hpu");
  auto res = t.lgamma_();

  EXPECT_EQ(t.storage().data_ptr().get(), res.storage().data_ptr().get());
}

// Test disabled since we do not want to support CPU Fallback for as_strided.
// Enable this test when strided tensors are completely supported on HPU and
// move it to appropriate test file
/*TEST_P(FallbackTest, NonSupportedAsStrided) {
  torch::Tensor A = torch::rand({3, 3, 3, 3, 3});
  torch::Tensor hA = A.to(torch::kHPU);
  at::Tensor Out = A.as_strided({2, 2}, {1, 2});
  at::Tensor hOut = hA.as_strided({2, 2}, {1, 2});

  Out.div_(4);
  hOut.div_(4);

  EXPECT_TRUE(allclose(Out, hOut.to("cpu"))) << Out << hOut.to("cpu");
}*/
