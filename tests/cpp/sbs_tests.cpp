/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
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
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/sbs_runner.h"

class SBSWithParamsTest
    : public ::testing::TestWithParam<std::tuple<habana_lazy::SBSModes, bool>>,
      public habana_lazy_test::EnvHelper {
  void SetUp() override {
    SetSeed();
    DisableCpuFallback();
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
  }

  void TearDown() override {
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    RestoreMode();
  }
};

TEST_P(SBSWithParamsTest, AddScalarSBS) {
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

TEST_P(SBSWithParamsTest, AddTensorsSBS) {
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

TEST_P(SBSWithParamsTest, MulSBS) {
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

TEST_P(SBSWithParamsTest, MulAddInplaceSBS) {
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

TEST_P(SBSWithParamsTest, AddInplaceSBS) {
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

TEST_P(SBSWithParamsTest, TopkSBSTest) {
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

TEST_P(SBSWithParamsTest, GraphTextDump1SBSTest) {
  int sbsType = std::get<0>(GetParam());
  SET_ENV_FLAG_NEW(PT_SBS, sbsType, 1);
  bool performMarkStep = std::get<1>(GetParam());
  std::cout << "PT_SBS=" << sbsType
            << " perform mark_step = " << performMarkStep << std::endl;
  auto A = torch::randn({2, 2}, torch::requires_grad(false));
  auto B = torch::randn({2, 2}, torch::requires_grad(false));
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto I = torch::add(hA, hB, 1.0);
  auto J = torch::relu(I);
  std::string string_J;
  if (performMarkStep) {
    auto hl_J = std::make_shared<habana_lazy::HbLazyTensor>(
        habana_lazy::GetHbLazyTensor(J));
    auto ir_value_J = hl_J->CurrentIrValue();
    if (ir_value_J.mp_node) {
      std::vector<habana_lazy::ir::NodePtr> a_J{ir_value_J.mp_node};
      string_J = habana_lazy::IrGraphDumpUtil::ToText(a_J);
    }

    habana_lazy::HbLazyTensor::StepMarker({});
  }
  auto out = torch::relu(J);

  auto hl_result = std::make_shared<habana_lazy::HbLazyTensor>(
      habana_lazy::GetHbLazyTensor(out));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<habana_lazy::ir::NodePtr> a{ir_value.mp_node};
  auto out_string = habana_lazy::IrGraphDumpUtil::ToText(a);

  if (performMarkStep) {
    EXPECT_EQ(
        string_J.find("IR {\n"
                      "  %0 = prim::constant(), value=1.\n"
                      "  %1 = hpu::input()\n"
                      "  %2 = hpu::input()\n"
                      "  %3 = aten::add(%2, %1, %0)\n"
                      "  %4 = aten::relu(%3), ROOT=0\n"
                      "}"),
        0);

    EXPECT_EQ(
        out_string.find("IR {\n"
                        "  %0 = hpu::input()\n"
                        "  %1 = aten::relu(%0), ROOT=0\n"
                        "}"),
        0);
  } else {
    EXPECT_EQ(
        out_string.find("IR {\n"
                        "  %0 = prim::constant(), value=1.\n"
                        "  %1 = hpu::input()\n"
                        "  %2 = hpu::input()\n"
                        "  %3 = aten::add(%2, %1, %0)\n"
                        "  %4 = aten::relu(%3)\n"
                        "  %5 = aten::relu(%4), ROOT=0\n"
                        "}"),
        0);
  }

  UNSET_ENV_FLAG_NEW(PT_SBS);
}

TEST_P(SBSWithParamsTest, CrossEntropySBSTest) {
  int sbsType = std::get<0>(GetParam());
  SET_ENV_FLAG_NEW(PT_SBS, sbsType, 1);
  bool performMarkStep = std::get<1>(GetParam());
  std::cout << "PT_SBS=" << sbsType
            << " perform mark_step = " << performMarkStep << std::endl;

  torch::Tensor input_tensor =
      torch::rand({64, 128, 48, 40}, torch::requires_grad(false));
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);

  torch::Tensor weight_tensor =
      torch::rand({4, 128, 1, 1}, torch::requires_grad(false));
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);
  if (!habana_lazy::exec::OptPassCfg::GetInstance()
           ->IsEnabledWeightPermutePass()) {
    auto wt_hwck = weight_tensor.permute({2, 3, 1, 0}).contiguous();
    tHabanaW = wt_hwck.to(torch::kHPU);
  }

  auto target = torch::randint(0, 3, {64, 48, 40}, torch::kLong);
  torch::Tensor htarget = target.to(torch::kHPU);

  torch::Tensor houtConv =
      torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::nn::CrossEntropyLoss loss;
  if (performMarkStep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto outhpu = loss->forward(houtConv, htarget);
  torch::Tensor out = outhpu.to(torch::kCPU);

  torch::Tensor outConv = torch::conv2d(
      input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto outcpu = loss->forward(outConv, target);

  EXPECT_EQ(allclose(out, outcpu, 0.001, 0.001), true);

  if (sbsType != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(outhpu);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));
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
    SBSWithParamsTest,
    ::testing::Combine(sbsTypes, performMarkStep));
