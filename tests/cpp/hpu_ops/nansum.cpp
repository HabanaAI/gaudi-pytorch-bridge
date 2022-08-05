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

TEST_F(HpuOpTest, nansum_4d_2d_keepdim) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  const std::vector<int64_t> dim{0, 2};

  auto expected = torch::nansum(GetCpuInput(0), dim, true /*keepdim*/);
  auto result = torch::nansum(GetHpuInput(0), dim, true /*keepdim*/);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum_3d_2d_keepdim_out) {
  GenerateInputs(1, {{5, 3, 6}});
  const std::vector<int64_t> dim{-2, 0};
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({1, 1, 6}, dtype);
  auto result =
      torch::empty({1, 1, 6}, torch::TensorOptions(dtype).device("hpu"));

  torch::nansum_outf(GetCpuInput(0), dim, true /*keepdim*/, dtype, expected);
  torch::nansum_outf(GetHpuInput(0), dim, true /*keepdim*/, dtype, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum_4d_4d_reduce_dim_out) {
  GenerateInputs(1, {{4, 6, 3, 2}});
  const std::vector<int64_t> dim{2, 1, -4, -1};
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::nansum_outf(GetCpuInput(0), dim, false /*keepdim*/, dtype, expected);
  torch::nansum_outf(GetHpuInput(0), dim, false /*keepdim*/, dtype, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum_4d_3d_reduce_dim) {
  GenerateInputs(1, {{3, 6, 5, 4}});
  const std::vector<int64_t> dim{3, 1, 0};

  auto expected = torch::nansum(GetCpuInput(0), dim, false /*keepdim*/);
  auto result = torch::nansum(GetHpuInput(0), dim, false /*keepdim*/);

  Compare(expected, result);
}

TEST_F(HpuOpTest, nansum) {
  GenerateInputs(1, {{3, 3, 4, 6}});

  auto expected = torch::nansum(GetCpuInput(0));
  auto result = torch::nansum(GetHpuInput(0));

  Compare(expected, result);
}
