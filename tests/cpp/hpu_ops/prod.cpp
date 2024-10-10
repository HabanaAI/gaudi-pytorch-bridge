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

TEST_F(HpuOpTest, prod_out) {
  GenerateInputs(1, {{3, 2, 2, 4}});
  torch::ScalarType dtype = torch::kFloat;
  int64_t dim = 3;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::prod_outf(GetCpuInput(0), dim, /*keepdim*/ true, dtype, expected);
  torch::prod_outf(GetHpuInput(0), dim, /*keepdim*/ true, dtype, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod1d) {
  GenerateInputs(1, {{2}});

  auto expected = torch::prod(GetCpuInput(0));
  auto result = torch::prod(GetHpuInput(0));
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_with_dtype2d) {
  GenerateInputs(1, {{2, 3}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::prod(GetCpuInput(0), dtype);
  auto result = torch::prod(GetHpuInput(0), dtype);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod3d) {
  GenerateInputs(1, {{2, 3, 4}});

  auto expected = torch::prod(GetCpuInput(0));
  auto result = torch::prod(GetHpuInput(0));
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_with_dtype4d) {
  GenerateInputs(1, {{2, 4, 2, 3}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::prod(GetCpuInput(0), dtype);
  auto result = torch::prod(GetHpuInput(0), dtype);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod5d) {
  GenerateInputs(1, {{1, 2, 3, 4, 5}});

  auto expected = torch::prod(GetCpuInput(0));
  auto result = torch::prod(GetHpuInput(0));
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_with_dtype6d) {
  GenerateInputs(1, {{1, 2, 2, 4, 5, 3}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::prod(GetCpuInput(0), dtype);
  auto result = torch::prod(GetHpuInput(0), dtype);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod7d) {
  GenerateInputs(1, {{2, 3, 2, 3, 4, 5, 1}});

  auto expected = torch::prod(GetCpuInput(0));
  auto result = torch::prod(GetHpuInput(0));
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_with_dtype8d) {
  GenerateInputs(1, {{3, 2, 1, 2, 2, 4, 5, 3}});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::prod(GetCpuInput(0), dtype);
  auto result = torch::prod(GetHpuInput(0), dtype);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_dim_with_dtype2d) {
  GenerateInputs(1, {{2, 3}});

  int64_t dim = -1;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::prod(GetCpuInput(0), dim, /*keepdim*/ true, dtype);
  auto result = torch::prod(GetHpuInput(0), dim, /*keepdim*/ true, dtype);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_dim3d) {
  GenerateInputs(1, {{3, 4, 5}});

  int64_t dim = 2;

  auto expected = torch::prod(GetCpuInput(0), dim, /*keepdim*/ false);
  auto result = torch::prod(GetHpuInput(0), dim, /*keepdim*/ false);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_dim_with_dtype4d) {
  GenerateInputs(1, {{1, 2, 2, 3}});

  int64_t dim = -2;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::prod(GetCpuInput(0), dim, /*keepdim*/ true, dtype);
  auto result = torch::prod(GetHpuInput(0), dim, /*keepdim*/ true, dtype);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_dim5d) {
  GenerateInputs(1, {{3, 2, 1, 2, 3}});

  int64_t dim = 3;

  auto expected = torch::prod(GetCpuInput(0), dim, /*keepdim*/ false);
  auto result = torch::prod(GetHpuInput(0), dim, /*keepdim*/ false);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_dim_with_dtype6d) {
  GenerateInputs(1, {{1, 2, 4, 1, 2, 3}});

  int64_t dim = -3;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::prod(GetCpuInput(0), dim, /*keepdim*/ true, dtype);
  auto result = torch::prod(GetHpuInput(0), dim, /*keepdim*/ true, dtype);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_dim7d) {
  GenerateInputs(1, {{1, 1, 2, 3, 4, 2, 3}});

  int64_t dim = 4;

  auto expected = torch::prod(GetCpuInput(0), dim, /*keepdim*/ false);
  auto result = torch::prod(GetHpuInput(0), dim, /*keepdim*/ false);
  Compare(expected, result);
}

TEST_F(HpuOpTest, prod_dim_with_dtype8d) {
  GenerateInputs(1, {{1, 2, 4, 5, 6, 1, 2, 3}});

  int64_t dim = -6;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::prod(GetCpuInput(0), dim, /*keepdim*/ true, dtype);
  auto result = torch::prod(GetHpuInput(0), dim, /*keepdim*/ true, dtype);
  Compare(expected, result);
}

// Check case for keepdim = false
TEST_F(HpuOpTest, prod_out_f) {
  GenerateInputs(1, {{4, 3, 3}});
  torch::ScalarType dtype = torch::kFloat;
  int64_t dim = -2;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::prod_outf(GetCpuInput(0), dim, /*keepdim*/ false, dtype, expected);
  torch::prod_outf(GetHpuInput(0), dim, /*keepdim*/ false, dtype, result);
  Compare(expected, result);
}