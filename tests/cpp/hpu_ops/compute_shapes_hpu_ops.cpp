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
class HpuOpComputeShapeTest : public HpuOpTestUtil {
  void SetUp() override {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, 1, 1);
  }
  void TearDown() override {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, 0, 1);
  }
};

TEST_F(HpuOpComputeShapeTest, bce_usual_3D_sum_cmptopshp) {
  const std::vector<int64_t> size = {8, 3, 2};
  GenerateInputs(3, {size, size, {8, 3, 1}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::binary_cross_entropy(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ GetCpuInput(1),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum);
  auto result = torch::binary_cross_entropy(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ GetHpuInput(1),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum);

  Compare(expected, result);
}

TEST_F(HpuOpComputeShapeTest, bce_usual_3D_sum_out_cmptopshp) {
  const std::vector<int64_t> size = {8, 3, 2};
  GenerateInputs(3, {size, size, {8, 3, 1}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty_like(GetCpuInput(0));
  auto result = torch::empty_like(GetHpuInput(0));
  expected = torch::binary_cross_entropy_outf(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ GetCpuInput(1),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum,
      expected);
  result = torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ GetHpuInput(1),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum,
      result);

  Compare(expected, result);
}
