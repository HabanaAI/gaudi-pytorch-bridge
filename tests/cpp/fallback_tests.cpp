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
#include "habana_lazy/debug_utils.h"

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

class FallbackWithParamsTest : public FallbackTest,
                               public ::testing::WithParamInterface<
                                   std::tuple<habana_lazy::SBSModes, bool>> {};

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

TEST_P(FallbackWithParamsTest, AddScalarSBS) {
  int sbsType = std::get<0>(GetParam());
  SET_ENV_FLAG_NEW(PT_SBS, sbsType, 1);
  bool performMarkStep = std::get<1>(GetParam());
  std::cout << "PT_SBS=" << sbsType
            << " perform mark_step = " << performMarkStep << std::endl;

  // HPU and SBS Run
  auto hpu_in =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kFloat));
  auto hpu_other =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kFloat));
  auto hpu_res = torch::add(hpu_in, hpu_other);
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res2 = torch::add(hpu_res, hpu_other);
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res3 = torch::add(hpu_res, hpu_res2);
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res4 = hpu_res3 + 5;
  auto hpu_res4_cpu = hpu_res4.to("cpu");

  if (sbsType != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_res4 = habana_lazy::GetHbLazyTensor(hpu_res4);
    c10::optional<at::Tensor> pTensor = hl_res4.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);

    auto hpu_res4_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(hpu_res4_cpu_ref, hpu_res4_cpu));
  }

  // CPU Run
  auto in =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kCPU).dtype(at::kFloat));
  auto other =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kCPU).dtype(at::kFloat));
  auto cpu_res = torch::add(in, other);
  auto cpu_res2 = torch::add(cpu_res, other);
  auto cpu_res3 = torch::add(cpu_res, cpu_res2);
  auto cpu_res4 = cpu_res3 + 5;
  EXPECT_TRUE(allclose(cpu_res4, hpu_res4_cpu));

  UNSET_ENV_FLAG_NEW(PT_SBS);
}

TEST_P(FallbackWithParamsTest, AddTensorsSBS) {
  int sbsType = std::get<0>(GetParam());
  SET_ENV_FLAG_NEW(PT_SBS, sbsType, 1);
  bool performMarkStep = std::get<1>(GetParam());
  std::cout << "PT_SBS=" << sbsType
            << " perform mark_step = " << performMarkStep << std::endl;

  // HPU and SBS Run
  auto hpu_in =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kFloat));
  auto hpu_other =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kFloat));
  auto hpu_res = torch::add(hpu_in, hpu_other);
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res2 = torch::add(hpu_res, hpu_other);
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res3 = hpu_res.add(hpu_res2);
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res3_cpu = hpu_res3.to("cpu");
  if (sbsType != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_res3 = habana_lazy::GetHbLazyTensor(hpu_res3);
    c10::optional<at::Tensor> pTensor = hl_res3.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);

    auto hpu_res3_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(hpu_res3_cpu_ref, hpu_res3_cpu));
  }

  // CPU Run
  auto in =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kCPU).dtype(at::kFloat));
  auto other =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kCPU).dtype(at::kFloat));
  auto cpu_res = torch::add(in, other);
  auto cpu_res2 = torch::add(cpu_res, other);
  auto cpu_res3 = cpu_res.add(cpu_res2);

  EXPECT_TRUE(allclose(cpu_res3, hpu_res3_cpu));

  UNSET_ENV_FLAG_NEW(PT_SBS);
}

TEST_P(FallbackWithParamsTest, MulSBS) {
  int sbsType = std::get<0>(GetParam());
  SET_ENV_FLAG_NEW(PT_SBS, sbsType, 1);
  bool performMarkStep = std::get<1>(GetParam());
  std::cout << "PT_SBS=" << sbsType
            << " perform mark_step = " << performMarkStep << std::endl;

  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});

  auto hA = A.to(torch::kHPU);
  auto exp = torch::mul(A, C);

  auto hC = C.to(torch::kHPU);
  auto result = torch::mul(hA, hC);
  torch::Tensor out = result.to(c10::kCPU);

  EXPECT_TRUE(allclose(out, exp, 0.001, 0.001));

  if (sbsType != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(result);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));
  }

  UNSET_ENV_FLAG_NEW(PT_SBS);
}

TEST_P(FallbackWithParamsTest, MulAddInplaceSBS) {
  int sbsType = std::get<0>(GetParam());
  SET_ENV_FLAG_NEW(PT_SBS, sbsType, 1);
  bool performMarkStep = std::get<1>(GetParam());
  std::cout << "PT_SBS=" << sbsType
            << " perform mark_step = " << performMarkStep << std::endl;

  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);

  A = A.add_(B);
  auto exp = torch::mul(A, C);

  hA = hA.add_(hB);
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto result = torch::mul(hA, hC);
  torch::Tensor out = result.to(c10::kCPU);

  EXPECT_TRUE(allclose(out, exp, 0.001, 0.001));

  if (sbsType != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(result);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));
  }

  UNSET_ENV_FLAG_NEW(PT_SBS);
}

TEST_P(FallbackWithParamsTest, AddInplaceSBS) {
  int sbsType = std::get<0>(GetParam());
  SET_ENV_FLAG_NEW(PT_SBS, sbsType, 1);
  bool performMarkStep = std::get<1>(GetParam());
  std::cout << "PT_SBS=" << sbsType
            << " perform mark_step = " << performMarkStep << std::endl;

  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);

  A = A.add_(B);
  auto exp = torch::add(A, C);

  hA = hA.add_(hB);
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto result = torch::add(hA, hC);
  // TODO: remove when this is resolved:
  // https://jira.habana-labs.com/browse/SW-69119
  habana_lazy::HbLazyTensor::StepMarker({});
  torch::Tensor out = result.to(c10::kCPU);

  EXPECT_TRUE(allclose(out, exp, 0.001, 0.001));

  if (sbsType != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(result);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));
  }

  UNSET_ENV_FLAG_NEW(PT_SBS);
}

TEST_P(FallbackWithParamsTest, TopkSBSTest) {
  int sbsType = std::get<0>(GetParam());
  SET_ENV_FLAG_NEW(PT_SBS, sbsType, 1);
  bool performMarkStep = std::get<1>(GetParam());
  std::cout << "PT_SBS=" << sbsType
            << " perform mark_step = " << performMarkStep << std::endl;

  auto self = torch::randn({3, 5});
  auto hself = self.to(torch::kHPU);

  auto out_cpu = at::topk(self, 2, 1, true, true);
  at::Tensor cout = std::get<0>(out_cpu);
  auto out_hpu = at::topk(hself, 2, 1, true, true);
  at::Tensor hout = std::get<0>(out_hpu).to(torch::kCPU);
  at::Tensor hout_hpu = std::get<0>(out_hpu);

  EXPECT_TRUE(cout.sizes().vec() == hout.sizes().vec());

  EXPECT_TRUE(allclose(cout, hout, 0.001, 0.001));

  if (sbsType != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(hout_hpu);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(hout, result_cpu_ref, 0.001, 0.001));
  }

  UNSET_ENV_FLAG_NEW(PT_SBS);
}

const auto sbsTypes = testing::Values(
    habana_lazy::SBS_MODE_DISABLED,
    habana_lazy::SBS_MODE_STANDALONE,
    habana_lazy::SBS_MODE_USE_CPU_INPUT,
    habana_lazy::SBS_MODE_USE_HPU_INPUT);

const auto performMarkStep = testing::Values(false, true);

INSTANTIATE_TEST_CASE_P(
    SBS,
    FallbackWithParamsTest,
    ::testing::Combine(sbsTypes, performMarkStep));

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

TEST_F(FallbackTest, inverse) {
  auto a = torch::randn({2, 2});
  auto b = a.inverse();
  auto out = torch::transpose(b, 0, 1);

  auto ha = a.to("hpu");
  auto hb = ha.inverse();
  auto hout = torch::transpose(hb, 0, 1);
  EXPECT_TRUE(allclose(out, hout.to("cpu"), 0.001, 0.001));
}
