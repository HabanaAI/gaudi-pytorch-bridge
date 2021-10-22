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

TEST_F(HpuOpTest, logsumexp_4d_3d_keepdim) {
  GenerateInputs(1, {{2, 3, 4, 5}}, {torch::kBFloat16});
  const std::vector<int64_t> dim{-2, -1, -4};

  auto expected = torch::logsumexp(GetCpuInput(0), dim, true);
  auto result = torch::logsumexp(GetHpuInput(0), dim, true);

  Compare(expected, result, 1e-03, 1e-02);
}

TEST_F(HpuOpTest, logsumexp_5d_4d_keepdim) {
  GenerateInputs(1, {{6, 7, 8, 9, 10}});
  const std::vector<int64_t> dim{0, 1, 3, 2};

  auto expected = torch::logsumexp(GetCpuInput(0), dim, true);
  auto result = torch::logsumexp(GetHpuInput(0), dim, true);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logsumexp_2d_1d) {
  GenerateInputs(1, {{4, 5}}, {torch::kBFloat16});
  const std::vector<int64_t> dim{0};

  auto expected = torch::logsumexp(GetCpuInput(0), dim, false);
  auto result = torch::logsumexp(GetHpuInput(0), dim, false);

  Compare(expected, result, 1e-03, 1e-02);
}

TEST_F(HpuOpTest, logsumexp_3d_2d) {
  GenerateInputs(1, {{12, 45, 61}});
  const std::vector<int64_t> dim{-3, -2};

  auto expected = torch::logsumexp(GetCpuInput(0), dim, false);
  auto result = torch::logsumexp(GetHpuInput(0), dim, false);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logsumexp_5d_2d_keepdim_out) {
  GenerateInputs(1, {{5, 10, 9, 6, 7}});
  const std::vector<int64_t> dim{-4, -1, -3};
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({5, 1, 1, 6, 1}, dtype);
  auto result =
      torch::empty({5, 1, 1, 6, 1}, torch::TensorOptions(dtype).device("hpu"));

  torch::logsumexp_outf(GetCpuInput(0), dim, true, expected);
  torch::logsumexp_outf(GetHpuInput(0), dim, true, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, logsumexp_4d_3d_out) {
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, {{2, 3, 4, 5}}, {dtype});
  const std::vector<int64_t> dim{0, 2, 3};

  auto expected = torch::empty((3), dtype);
  auto result = torch::empty((3), torch::TensorOptions(dtype).device("hpu"));

  torch::logsumexp_outf(GetCpuInput(0), dim, false, expected);
  torch::logsumexp_outf(GetHpuInput(0), dim, false, result);

  Compare(expected, result, 1e-03, 1e-02);
}
