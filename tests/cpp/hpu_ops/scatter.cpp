/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

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
