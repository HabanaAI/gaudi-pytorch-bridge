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
#include "habana_lazy/sbs_debug.h"
#include "habana_lazy/sbs_runner.h"

class SBSWithParamsTest
    : public ::testing::TestWithParam<std::tuple<habana_lazy::SBSModes, bool>>,
      public habana_lazy_test::EnvHelper {
  void SetUp() override {
    SetSeed();
    DisableCpuFallback();
    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();

    m_sbs_mode = std::get<0>(GetParam());
    SET_ENV_FLAG_NEW(PT_SBS, m_sbs_mode, 1);
    m_perform_markstep = std::get<1>(GetParam());
    std::cout << "PT_SBS=" << m_sbs_mode
              << " perform mark_step = " << m_perform_markstep << std::endl;

    m_numberOfPotentialSBSOps = 0;
    m_numberOfPotentialSBSOpTensors = 0;
    m_numberOfCopiesToHPU = 0; // copy is not a lazy op, but it is aggregated
                               // in the value GetNumberOfAccumulatedOps
  }

  void TearDown() override {
    habana_lazy::SBSDebug::getInstance().reset();
    habana_lazy::SBSInterface::reset();
    UNSET_ENV_FLAG_NEW(PT_SBS);

    habana_lazy::exec::OptPassCfg::GetInstance()->SetDefaultOptFlags();
    RestoreMode();
  }

 protected:
  int m_sbs_mode = habana_lazy::SBS_MODE_DISABLED;
  bool m_perform_markstep = false;

  // count validation
  size_t m_numberOfPotentialSBSOps = 0;
  size_t m_numberOfPotentialSBSOpTensors = 0;
  size_t m_numberOfCopiesToHPU =
      0; // copy is not a lazy op, but it is aggregated
         // in the value GetNumberOfAccumulatedOps

  void PerformMarkStep() {
    if (m_perform_markstep) {
      PT_TEST_DEBUG("Calling StepMarker");
      habana_lazy::HbLazyTensor::StepMarker({});
    }
  }

  void UpdateOpCounters() {
    m_numberOfPotentialSBSOps +=
        habana_lazy::SBSDebug::getInstance().GetNumberOfAccumulatedOps();
    m_numberOfPotentialSBSOpTensors +=
        habana_lazy::SBSDebug::getInstance()
            .GetNumberOfAccumulatedOpOutputTensors();
  }

  void ValidateCounters() {
    // in lazy1 we might skip comparing middle-graph tensors, as we don't sync
    // all of the tensors. This includes inplace tensors.
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1) {
      m_numberOfCopiesToHPU +=
          habana_lazy::SBSInterface::getNumberOfTensorCopies();
      // subtracting the copy_to_hpu op
      m_numberOfPotentialSBSOps -= m_numberOfCopiesToHPU;
      m_numberOfPotentialSBSOpTensors -= m_numberOfCopiesToHPU;
      EXPECT_EQ(
          m_numberOfPotentialSBSOps,
          habana_lazy::SBSInterface::getNumberOfHandledOps() +
              habana_lazy::SBSInterface::getNumberOfOpTries());

      EXPECT_EQ(
          m_numberOfPotentialSBSOpTensors,
          habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    }
    // in lazy2 we run each op in separate graphs, so the above count is
    // irrelevant
    else if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
      EXPECT_EQ(
          habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
          habana_lazy::SBSInterface::getNumberOfHandledOpTensors());
    }
  }

  void ConvolutionSBSTest(bool channelLast, bool random);
};

TEST_P(SBSWithParamsTest, AddScalarSBS) {
  // HPU and SBS Run
  auto hpu_in =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kFloat));
  auto hpu_other =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kFloat));
  auto hpu_res = torch::add(hpu_in, hpu_other);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res2 = torch::add(hpu_res, hpu_other);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res3 = torch::add(hpu_res, hpu_res2);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res4 = hpu_res3 + 5;
  auto hpu_res4_cpu = hpu_res4.to("cpu");

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_res4 = habana_lazy::GetHbLazyTensor(hpu_res4);
    c10::optional<at::Tensor> pTensor = hl_res4.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);

    auto hpu_res4_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(hpu_res4_cpu_ref, hpu_res4_cpu));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
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
}

TEST_P(SBSWithParamsTest, AddTensorsSBS) {
  // HPU and SBS Run
  auto hpu_in =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kFloat));
  auto hpu_other =
      torch::tensor({{1, 2}, {3, 4}}, at::device(at::kHPU).dtype(at::kFloat));
  auto hpu_res = torch::add(hpu_in, hpu_other);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res2 = torch::add(hpu_res, hpu_other);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res3 = hpu_res.add(hpu_res2);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto hpu_res3_cpu = hpu_res3.to("cpu");
  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_res3 = habana_lazy::GetHbLazyTensor(hpu_res3);
    c10::optional<at::Tensor> pTensor = hl_res3.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);

    auto hpu_res3_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(hpu_res3_cpu_ref, hpu_res3_cpu));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
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
}

TEST_P(SBSWithParamsTest, MulSBS) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});

  auto hA = A.to(torch::kHPU);
  auto exp = torch::mul(A, C);

  auto hC = C.to(torch::kHPU);
  auto result = torch::mul(hA, hC);
  torch::Tensor out = result.to(c10::kCPU);

  EXPECT_TRUE(allclose(out, exp, 0.001, 0.001));

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(result);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }
}

TEST_P(SBSWithParamsTest, MulAddInplaceSBS) {
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
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto result = torch::mul(hA, hC);
  torch::Tensor out = result.to(c10::kCPU);

  EXPECT_TRUE(allclose(out, exp, 0.001, 0.001));

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(result);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }
}

TEST_P(SBSWithParamsTest, AddInplaceSBS) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});

  auto hA = A.to(torch::kHPU);
  m_numberOfCopiesToHPU++;
  auto hB = B.to(torch::kHPU);
  m_numberOfCopiesToHPU++;
  auto hC = C.to(torch::kHPU);
  m_numberOfCopiesToHPU++;

  A = A.add_(B);
  A = A.add_(B);
  auto exp = torch::add(A, C);

  hA = hA.add_(hB);
  PerformMarkStep();
  hA = hA.add_(hB);
  auto result = torch::add(hA, hC);
  UpdateOpCounters();
  torch::Tensor out = result.to(c10::kCPU);

  EXPECT_TRUE(allclose(out, exp, 0.001, 0.001));

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(result);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

    ValidateCounters();

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }
}

TEST_P(SBSWithParamsTest, TopkSBSTest) {
  auto self = torch::randn({3, 5});
  auto hself = self.to(torch::kHPU);

  auto out_cpu = at::topk(self, 2, 1, true, true);
  at::Tensor cout = std::get<0>(out_cpu);
  auto out_hpu = at::topk(hself, 2, 1, true, true);
  at::Tensor hout = std::get<0>(out_hpu).to(torch::kCPU);
  at::Tensor hout_hpu = std::get<0>(out_hpu);

  EXPECT_TRUE(cout.sizes().vec() == hout.sizes().vec());

  EXPECT_TRUE(allclose(cout, hout, 0.001, 0.001));

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(hout_hpu);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(hout, result_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }
}

// Test currently crashes. See here for updates: [SW-75234]
TEST_P(SBSWithParamsTest, DISABLED_GraphTextDump1SBSTest) {
  auto A = torch::randn({2, 2}, torch::requires_grad(false));
  auto B = torch::randn({2, 2}, torch::requires_grad(false));
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto I = torch::add(hA, hB, 1.0);
  auto J = torch::relu(I);
  std::string string_J;
  if (m_perform_markstep) {
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

  std::cout << out_string << std::endl;

  if (m_perform_markstep) {
    std::cout << string_J << std::endl;
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
}

TEST_P(SBSWithParamsTest, CrossEntropySBSTest) {
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
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});
  auto outhpu = loss->forward(houtConv, htarget);
  torch::Tensor out = outhpu.to(torch::kCPU);

  torch::Tensor outConv = torch::conv2d(
      input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  auto outcpu = loss->forward(outConv, target);

  EXPECT_EQ(allclose(out, outcpu, 0.001, 0.001), true);

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(outhpu);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

    // currently failing, see this: [SW-75400]
    // EXPECT_EQ(
    //     habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
    //     habana_lazy::SBSInterface::getNumberOfHandledOpTensors());
  }
}

void SBSWithParamsTest::ConvolutionSBSTest(bool channelLast, bool random) {
  PT_TEST_DEBUG(
      "Test memory format: ", channelLast ? "channels last" : "contiguous");
  c10::MemoryFormat format = channelLast ? c10::MemoryFormat::ChannelsLast
                                         : c10::MemoryFormat::Contiguous;
  torch::Tensor input_tensor = random
      ? torch::rand({1, 2, 4, 4}, torch::requires_grad(false)).to(format)
      : torch::tensor(
            {{
                {{1, 2, 3, 4}, {-5, -6, -7, -8}, {9, 10, 11, 12}},
                {{-1, -2, -3, -4}, {5, 6, 7, 8}, {-9, -10, -11, -12}},
            }},
            torch::dtype(torch::kFloat).requires_grad(false))
            .to(format);
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU).contiguous(format);

  torch::Tensor weight_tensor = random
      ? torch::rand({2, 2, 3, 3}, torch::requires_grad(false))
      : torch::tensor(
            {{{{1, 2, 3}, {-4, -5, -6}, {7, 8, 9}},
              {{-1, -2, -3}, {4, 5, 6}, {-7, -8, -9}}},
             {{{10, 20, 30}, {-40, -50, -60}, {70, 80, 90}},
              {{-10, -20, -30}, {40, 50, 60}, {-70, -80, -90}}}},
            torch::dtype(torch::kFloat).requires_grad(false));
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);
  if (!habana_lazy::exec::OptPassCfg::GetInstance()
           ->IsEnabledWeightPermutePass()) {
    auto wt_hwck = weight_tensor.permute({2, 3, 1, 0}).contiguous();
    tHabanaW = wt_hwck.to(torch::kHPU);
  }

  torch::Tensor houtConv =
      torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
  torch::Tensor out = houtConv.to(torch::kCPU);

  torch::Tensor outConv = torch::conv2d(
      input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  EXPECT_EQ(allclose(out, outConv, 0.001, 0.001), true);

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(houtConv);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }
}

TEST_P(SBSWithParamsTest, ConvolutionSBSTest_random_CL) {
  ConvolutionSBSTest(true, true);
}
TEST_P(SBSWithParamsTest, ConvolutionSBSTest_const_CL) {
  ConvolutionSBSTest(true, false);
}
TEST_P(SBSWithParamsTest, ConvolutionSBSTest_random_Contiguous) {
  ConvolutionSBSTest(false, true);
}
TEST_P(SBSWithParamsTest, ConvolutionSBSTest_const_Contiguous) {
  ConvolutionSBSTest(false, false);
}

// Graph :
//
//     Bias1  Bias2           Data
//       \    /               |
//        \  /                |
//         Add                |
//          |-(weights)->  Convolution 3x3
//                            |
//                        Batch Norm
//                        Max Pool 2D           Bias3
//                           Relu             Broadcast
//                            |                  |
//                           Add <----------------
//                            |
//                         view/Reshape
//                            |
//                       UpSampleNearest2d
//                            |
//                           Add.Scalar(Constant)
//                            |
//                           out

TEST_P(SBSWithParamsTest, DynamicShapeSBSTest4) {
  int kH = 3;
  int kW = 3;
  const int C = 16;
  const int N = 16;
  int H = 16;
  at::Scalar inScalar = 2.0;
  std::vector<int> in_sizes{16, 32, 64};
  for (int i = 0; i < in_sizes.size(); i++) {
    PT_TEST_DEBUG("PTI_DBG: Iteration Start -- ", i, " ----\n");
    int W = in_sizes[i];
    // weight_tensor = bias1 + bias2
    torch::Tensor bias1 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor bias2 =
        torch::randn({C, C, kW, kH}, torch::requires_grad(false));
    torch::Tensor h_bias1 = bias1.to(torch::kHPU);
    torch::Tensor h_bias2 = bias2.to(torch::kHPU);
    torch::Tensor weight_tensor = torch::add(bias1, bias2);
    torch::Tensor h_weight_tensor = torch::add(h_bias1, h_bias2);
    // out_conv = Conv3x3(Data, weight)
    torch::Tensor in_tensor =
        torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor h_in_tensor = in_tensor.to(torch::kHPU);
    torch::Tensor h_weight_tensor_hwck = h_weight_tensor;
    if (!habana_lazy::exec::OptPassCfg::GetInstance()
             ->IsEnabledWeightPermutePass()) {
      h_weight_tensor_hwck = h_weight_tensor.permute({2, 3, 1, 0}).contiguous();
    }
    torch::Tensor h_out_conv = torch::conv2d(
        h_in_tensor, h_weight_tensor_hwck, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor out_conv = torch::conv2d(
        in_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    // bn_out = BatchNorm(out_conv)
    torch::Tensor gamma =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor beta =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor mean =
        torch::randn(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor var =
        torch::ones(C, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_gamma = gamma.to(torch::kHPU);
    torch::Tensor h_beta = beta.to(torch::kHPU);
    torch::Tensor h_mean = mean.to(torch::kHPU);
    torch::Tensor h_var = var.to(torch::kHPU);
    float mom = 0.1;
    float eps = 1e-5;
    auto h_bn_outs = torch::native_batch_norm(
        h_out_conv, h_gamma, h_beta, h_mean, h_var, false, mom, eps);
    auto bn_outs = torch::native_batch_norm(
        out_conv, gamma, beta, mean, var, false, mom, eps);
    auto h_bn_out = std::get<0>(h_bn_outs);
    auto bn_out = std::get<0>(bn_outs);
    // pool_out = MaxPool2D(bn_out)
    auto h_pool_outs = torch::max_pool2d_with_indices(
        h_bn_out, {2, 2}, {2, 2}, {0, 0}, {1, 1}, true);
    torch::Tensor h_pool_out = std::get<0>(h_pool_outs);
    torch::Tensor pool_out = torch::max_pool2d(bn_out, 2, 2);
    if (m_perform_markstep)
      habana_lazy::HbLazyTensor::StepMarker({});
    // relu_out = relu(pool_out)
    torch::Tensor h_relu_out = torch::relu(h_pool_out);
    torch::Tensor relu_out = torch::relu(pool_out);
    // out = add(relu_out, x)
    torch::Tensor bias3 =
        torch::randn(1, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor h_bias3 = bias3.to(torch::kHPU);
    auto h_out_add = torch::add(h_relu_out, h_bias3);
    auto out_add = torch::add(relu_out, bias3);
    // out = upsample(out_add,2)
    std::array<double, 2> scale_array = {2.0, 2.0};
    c10::ArrayRef<double> scale_factors = scale_array;
    auto h_out_upsample =
        torch::upsample_nearest2d(h_out_add, {}, scale_factors);
    auto out_upsample = torch::upsample_nearest2d(out_add, {}, scale_factors);
    // out = view(out_upsample)
    auto h_out_view = h_out_upsample.view({-1});
    auto out_view = out_upsample.view({-1});
    // out = Add(out_view,2)
    auto h_out = torch::add(h_out_view, inScalar);
    auto out = torch::add(out_view, inScalar);

    torch::Tensor out_hpu = h_out.to(torch::kCPU);
    EXPECT_EQ(allclose(out_hpu, out, 0.01, 0.01), true);

    // TODO: add comare tests inside the process
    if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
      auto hl_result = habana_lazy::GetHbLazyTensor(h_out);
      c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
      ASSERT_NE(pTensor, c10::nullopt);
      auto result_cpu_ref = pTensor.value();

      EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

      EXPECT_EQ(
          habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
          habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

      // Fix errors and restore
      // EXPECT_EQ(
      //     habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
      // EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
    }

    PT_TEST_DEBUG("PTI_DBG: Iteration End -- ", i, " ----\n");
  }
}

#if 0
TEST_P(SBSWithParamsTest, stridedinsertreuseSBS) {
  torch::Tensor A = torch::randn({4});
  auto b = torch::relu(A);
  auto v1 = A.view(-1);
  auto grad1 = torch::randn({4});

  auto hA = A.to(torch::kHPU);
  auto hB = torch::relu(hA);
  auto hv1 = hA.view(-1);
  auto hgrad1 = grad1.to(torch::kHPU);

  v1.mul_(grad1);

  hv1.mul_(hgrad1);

  habana_lazy::HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(hv1);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(v1, result_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());
  }


}
#endif

TEST_P(SBSWithParamsTest, AddViewSBSTest) {
  int N = 1;
  int C = 2;
  int H = 4;
  at::Scalar alpha = 0.5;
  at::Scalar Y = 2.0;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor C = A.view(-1);
    torch::Tensor out_cpu = C.add(alpha);
    torch::Tensor hC = hA.view(-1);
    torch::Tensor out_hpu = hC.add(alpha);
    auto out = out_hpu.to(torch::kCPU);
    EXPECT_EQ(allclose(out, out_cpu, 0.001, 0.001), true);

    if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
      auto hl_result = habana_lazy::GetHbLazyTensor(out_hpu);
      c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
      ASSERT_NE(pTensor, c10::nullopt);
      auto result_cpu_ref = pTensor.value();

      EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

      EXPECT_EQ(
          habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
          habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

      EXPECT_EQ(
          habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
      EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
    }
  }
}

// Test currently fails in accuracy / CPU existence (depends on lazy mode).
// see updates here: [SW-75044][SW-72745]
TEST_P(SBSWithParamsTest, DISABLED_AddInplaceViewSBSTest) {
  int N = 1;
  int C = 2;
  int H = 4;
  at::Scalar alpha = 0.5;
  at::Scalar Y = 2.0;
  std::vector<int> in_sizes{8, 10, 12, 20};
  for (int i = 0; i < in_sizes.size(); i++) {
    int W = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({N, C, H, W}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor C = A.view(-1);
    torch::Tensor out_cpu = C.add_(alpha);
    torch::Tensor hC = hA.view(-1);
    torch::Tensor out_hpu = hC.add_(alpha);
    auto out = out_hpu.to(torch::kCPU);
    EXPECT_EQ(allclose(out, out_cpu, 0.001, 0.001), true);

    if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
      auto hl_result = habana_lazy::GetHbLazyTensor(out_hpu);
      c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
      ASSERT_NE(pTensor, c10::nullopt);
      auto result_cpu_ref = pTensor.value();

      EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

      EXPECT_EQ(
          habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
          habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

      EXPECT_EQ(
          habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
      EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
    }
  }
}

TEST_P(SBSWithParamsTest, ViewsTestSBS) {
  torch::Tensor A = torch::rand({3, 3, 3, 3, 3}, torch::kFloat);
  std::cout << "A.dtype(): " << A.dtype() << std::endl;
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor B = A.add(1);
  torch::Tensor hB = hA.add(1);
  at::Tensor out = B.view({3, 3, 3, 3, 3});
  std::cout << "out.dtype(): " << out.dtype() << std::endl;
  at::Tensor hout = hB.view({3, 3, 3, 3, 3});

  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});

  out = out.div(4);
  hout = hout.div(4);

  EXPECT_TRUE(allclose(out, hout.to(c10::kCPU)))
      << "out: " << out << " hout: " << hout.to(c10::kCPU);

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(hout);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

    auto hl_B = habana_lazy::GetHbLazyTensor(hB);
    c10::optional<at::Tensor> pTensorB = hl_B.GetCPUTensorData();
    ASSERT_NE(pTensorB, c10::nullopt);
    auto B_cpu_ref = pTensorB.value();
    EXPECT_TRUE(allclose(B, B_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }

  EXPECT_TRUE(allclose(B, hB.to(c10::kCPU)))
      << "B: " << B << " hB: " << hB.to(c10::kCPU);
}

TEST_P(SBSWithParamsTest, AddTensorsViewsSBS) {
  auto in = torch::randint(-100, 100, {3, 3, 3, 3, 3}, torch::kFloat);
  auto hpu_in = in.to(c10::kHPU);

  auto other = torch::randint(-100, 100, {3, 3, 3, 3, 3}, torch::kFloat);
  auto hpu_other = other.to(c10::kHPU);

  auto res = torch::add(in, other);
  auto hpu_res = torch::add(hpu_in, hpu_other);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});

  auto res2 = torch::add(res, other);
  auto hpu_res2 = torch::add(hpu_res, hpu_other);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});

  auto res_view = res.view({3, 3, 3, 3, 3});
  auto hpu_res_view = hpu_res.view({3, 3, 3, 3, 3});
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});

  res_view = res_view.sub(in);
  hpu_res_view = hpu_res_view.sub(in);

  auto res3 = res.add(res2);
  auto hpu_res3 = hpu_res.add(hpu_res2);
  if (m_perform_markstep)
    habana_lazy::HbLazyTensor::StepMarker({});

  auto hpu_res3_cpu = hpu_res3.to(c10::kCPU);
  EXPECT_TRUE(allclose(res3, hpu_res3_cpu));

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_res3 = habana_lazy::GetHbLazyTensor(hpu_res3);
    c10::optional<at::Tensor> pTensor = hl_res3.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);

    auto hpu_res3_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(hpu_res3_cpu_ref, hpu_res3_cpu));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }
}

#if 0 // we need to solve view issues
TEST_P(SBSWithParamsTest, ViewsInplaceTestSBS) {
  // c10::get_backtrace()

  torch::Tensor A = torch::rand({3, 3, 3, 3, 3}, torch::kFloat);
  std::cout << "A.dtype(): " << A.dtype() << std::endl;
  torch::Tensor hA = A.to(torch::kHPU);
  at::Tensor out = A.view({3, 3, 3, 3, 3});
  std::cout << "out.dtype(): " << out.dtype() << std::endl;
  at::Tensor hout = hA.view({3, 3, 3, 3, 3});

  out.div_(4);
  hout.div_(4);

  EXPECT_TRUE(allclose(out, hout.to(c10::kCPU)))
      << "out: " << out << " hout: " << hout.to(c10::kCPU);

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(hout);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();
    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());
  }


}
#endif

#if 0 // TODO 
TEST_P(SBSWithParamsTest, GraphTextDumpBCESBSTest) {
  auto input = torch::randn({6, 1}, at::requires_grad());
  auto target = torch::randn({6, 1}); // Nx1
  auto grad_output = torch::randn({1});
  auto wt = torch::randn({6, 1});

  torch::Tensor hinput = input.to(torch::kHPU).detach();
  hinput.set_requires_grad(true);
  torch::Tensor htarget = target.to(torch::kHPU);
  torch::Tensor hgrad_out = grad_output.to(torch::kHPU);
  torch::Tensor hwt = wt.to(torch::kHPU);

  // auto hsigmout = torch::sigmoid(hinput);
  auto houtput =
      torch::binary_cross_entropy(torch::sigmoid(hinput), htarget, hwt,
      at::Reduction::Mean);
  auto hboutput = torch::binary_cross_entropy_backward(
      hgrad_out, torch::sigmoid(hinput), htarget, hwt, at::Reduction::Mean);

  auto hl_result = std::make_shared<habana_lazy::HbLazyTensor>(
      habana_lazy::GetHbLazyTensor(hboutput));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<habana_lazy::ir::NodePtr> a{ir_value.mp_node};
  auto out_string = habana_lazy::IrGraphDumpUtil::ToText(a);

  auto houtfwd = houtput.to(torch::kCPU);
  auto houtbwd = hboutput.to(torch::kCPU);

  // reference output
  auto expfwd = torch::binary_cross_entropy(
      torch::sigmoid(input), target, wt, at::Reduction::Mean);
  auto expbwd = torch::binary_cross_entropy_backward(
      grad_output, torch::sigmoid(input), target, wt, at::Reduction::Mean);

  std::cout << "out_string: " << out_string << std::endl;

  EXPECT_EQ(allclose(houtfwd, expfwd), true);
  EXPECT_EQ(allclose(houtbwd, expbwd), true);
}
#endif

TEST_P(SBSWithParamsTest, MaxPoolBWDSBSTest) {
  auto input_tensor =
      torch::arange(20, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 4, 5}); // nchw
  auto cpu_pool = torch::max_pool2d(input_tensor, 3, 1);
  auto cpu_out = torch::relu(cpu_pool);

  // fwd propga
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  m_numberOfCopiesToHPU++;
  auto outHabana1 = torch::max_pool2d_with_indices(
      tHabanaX, {3, 3}, {1, 1}, {0, 0}, {1, 1}, true);
  torch::Tensor outHabana = torch::relu(std::get<0>(outHabana1));

  // bwd propga with dummy grad tensor
  auto grad_tensor =
      torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 2, 3});
  torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
  m_numberOfCopiesToHPU++;
  PerformMarkStep();
  outHabana.backward({tHabanaG}, false, true);

  UpdateOpCounters();
  auto out_cpu_lazy = outHabana.to(torch::kCPU);
  ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(outHabana);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(cpu_out, result_cpu_ref, 0.001, 0.001));

    ValidateCounters();

    // Fix errors and restore
    // EXPECT_EQ(
    //     habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    // EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }
}

// Dimensions don't match 4/5d, hence disabled
TEST_P(SBSWithParamsTest, DISABLED_PermuteSBSTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.permute({1, 0});
  torch::Tensor Out = A.permute({1, 0});

  PT_TEST_DEBUG("HPU: ", hOut);
  PT_TEST_DEBUG("CPU: ", Out);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(hOut);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(Out, result_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());
  }
}

TEST_P(SBSWithParamsTest, permuteSBSTest2) {
  torch::Tensor A = torch::randn({5, 6, 24, 24});
  torch::Tensor hA = A.to(torch::kHPU);
  auto hOut = hA.permute({0, 2, 3, 1});
  auto out = A.permute({0, 2, 3, 1});
  auto hOut_cpu = hOut.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);

  if (m_sbs_mode != habana_lazy::SBS_MODE_DISABLED) {
    auto hl_result = habana_lazy::GetHbLazyTensor(hOut);
    c10::optional<at::Tensor> pTensor = hl_result.GetCPUTensorData();
    ASSERT_NE(pTensor, c10::nullopt);
    auto result_cpu_ref = pTensor.value();

    EXPECT_TRUE(allclose(out, result_cpu_ref, 0.001, 0.001));

    EXPECT_EQ(
        habana_lazy::SBSDebug::getInstance().GetNumberOfReportLines(),
        habana_lazy::SBSInterface::getNumberOfHandledOpTensors());

    EXPECT_EQ(habana_lazy::SBSDebug::getInstance().GetNumberOfErrorLines(), 0);
    EXPECT_EQ(habana_lazy::SBSInterface::getNumberOfErrors(), 0);
  }
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
