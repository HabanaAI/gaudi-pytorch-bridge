/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest-param-test.h>
#include "habana_kernels/fallback_helper.h"

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, scatter) {
  GenerateInputs(2, {{4, 4}, {4, 4}}, {torch::kFloat, torch::kFloat});
  auto self_cpu = GetCpuInput(0);
  auto self_hpu = GetHpuInput(0);
  auto src_cpu = GetCpuInput(1);
  auto src_hpu = GetHpuInput(1);

  GenerateIntInputs(1, {{1, 4}}, 0, 4);
  auto indices_cpu = GetCpuInput(0).to(torch::kLong);
  auto indices_hpu = GetHpuInput(0).to(torch::kLong);

  auto expected = torch::scatter(self_cpu, 0, indices_cpu, src_cpu);
  auto result = torch::scatter(self_hpu, 0, indices_hpu, src_hpu);
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, scatter_inplace) {
  GenerateInputs(2, {{4, 4}, {4, 4}}, {torch::kFloat, torch::kFloat});
  auto self_cpu = GetCpuInput(0);
  auto self_hpu = GetHpuInput(0);
  auto src_cpu = GetCpuInput(1);
  auto src_hpu = GetHpuInput(1);

  GenerateIntInputs(1, {{1, 4}}, 0, 4);
  auto indices_cpu = GetCpuInput(0).to(torch::kLong);
  auto indices_hpu = GetHpuInput(0).to(torch::kLong);

  self_cpu.scatter_(0, indices_cpu, src_cpu);
  self_hpu.scatter_(0, indices_hpu, src_hpu);
  Compare(self_cpu, self_hpu, 0, 0);
}

TEST_F(HpuOpTest, scatter_val_inplace) {
  GenerateInputs(1, {{4, 4}}, {torch::kInt});
  auto self_cpu = GetCpuInput(0);
  auto self_hpu = GetHpuInput(0);
  float val = 0.123;

  GenerateIntInputs(1, {{1, 4}}, 0, 4);
  auto indices_cpu = GetCpuInput(0).to(torch::kLong);
  auto indices_hpu = GetHpuInput(0).to(torch::kLong);

  self_cpu.scatter_(0, indices_cpu, val);
  self_hpu.scatter_(0, indices_hpu, val);
  Compare(self_cpu, self_hpu, 0, 0);
}

TEST_F(HpuOpTest, scatter_byte) {
  GenerateInputs(2, {{4, 4}, {4, 4}}, {torch::kByte, torch::kByte});
  auto self_cpu = GetCpuInput(0);
  auto self_hpu = GetHpuInput(0);
  auto src_cpu = GetCpuInput(1);
  auto src_hpu = GetHpuInput(1);

  GenerateIntInputs(1, {{1, 4}}, 0, 4);
  auto indices_cpu = GetCpuInput(0).to(torch::kLong);
  auto indices_hpu = GetHpuInput(0).to(torch::kLong);
  auto expected = torch::scatter(self_cpu, 0, indices_cpu, src_cpu);
  auto result = torch::scatter(self_hpu, 0, indices_hpu, src_hpu);
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, scatter_out) {
  GenerateInputs(
      3,
      {/*self*/ {4, 4}, /*src*/ {4, 4}, /*output*/ {4, 4}},
      {torch::kBFloat16, torch::kBFloat16, torch::kBFloat16});
  auto self_cpu = GetCpuInput(0);
  auto self_hpu = GetHpuInput(0);
  auto src_cpu = GetCpuInput(1);
  auto src_hpu = GetHpuInput(1);
  auto out_cpu = GetCpuInput(2);
  auto out_hpu = GetHpuInput(2);

  GenerateIntInputs(1, {{1, 4}}, 0, 4);
  auto indices_cpu = GetCpuInput(0).to(torch::kLong);
  auto indices_hpu = GetHpuInput(0).to(torch::kLong);

  torch::scatter_outf(self_cpu, 0, indices_cpu, src_cpu, out_cpu);
  torch::scatter_outf(self_hpu, 0, indices_hpu, src_hpu, out_hpu);
  Compare(out_cpu, out_hpu, 0, 0);
}

TEST_F(HpuOpTest, scatter_out_bool_val) {
  GenerateInputs(
      2, {/*self*/ {4, 4}, /*output*/ {4, 4}}, {torch::kBool, torch::kBool});
  auto self_cpu = GetCpuInput(0);
  auto self_hpu = GetHpuInput(0);
  auto out_cpu = GetCpuInput(1);
  auto out_hpu = GetHpuInput(1);

  GenerateIntInputs(1, {{1, 4}}, 0, 4);
  auto indices_cpu = GetCpuInput(0).to(torch::kLong);
  auto indices_hpu = GetHpuInput(0).to(torch::kLong);

  float val = 0.4;

  torch::scatter_outf(self_cpu, 0, indices_cpu, val, out_cpu);
  torch::scatter_outf(self_hpu, 0, indices_hpu, val, out_hpu);
  Compare(out_cpu, out_hpu, 0, 0);
}

class ScatterDTypeSupportTest
    : public DTypeSupportTest<std::tuple<c10::ScalarType, c10::ScalarType>> {};

TEST_P(ScatterDTypeSupportTest, ScatterValueOut) {
  auto tensor_dtype = std::get<0>(GetParam());
  auto index_dtype = std::get<1>(GetParam());
  auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kHPU);
  auto input = torch::tensor({{1, 2, 3, 4}, {1, 2, 3, 4}}, options);
  auto output = torch::clone(input);
  auto indices = torch::tensor({1, 3}, options.dtype(index_dtype));

  torch::scatter_out(output, input, 0, indices, 8).to(torch::kCPU);
  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(
      op_fallback_frequency.find("aten::scatter.value_out"),
      op_fallback_frequency.end());
}

INSTANTIATE_TEST_SUITE_P(
    ScatterValueOutFallback,
    ScatterDTypeSupportTest,
    testing::Combine(
        testing::Values(
            torch::kFloat32,
            torch::kBFloat16,
            torch::kInt32,
            torch::kInt8,
            torch::kUInt8),
        testing::Values(torch::kInt32, torch::kInt64)));
