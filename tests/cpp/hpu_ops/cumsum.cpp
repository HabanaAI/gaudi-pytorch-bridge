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
#define SIZE(...) __VA_ARGS__

#define HPU_CUMSUM_TEST(op, in_size, dimension, datatype)            \
  TEST_F(HpuOpTest, op) {                                            \
    torch::ScalarType dtype = datatype;                              \
    GenerateInputs(1, {in_size}, {dtype});                           \
    auto expected = torch::cumsum(GetCpuInput(0), dimension, dtype); \
    auto result = torch::cumsum(GetHpuInput(0), dimension, dtype);   \
    Compare(expected, result);                                       \
  }

class HpuOpTest : public HpuOpTestUtil {};
HPU_CUMSUM_TEST(cumsum_f32, SIZE({2, 3}), 1, torch::kFloat)
HPU_CUMSUM_TEST(cumsum_bf16, SIZE({16, 8, 4}), -3, torch::kBFloat16)
HPU_CUMSUM_TEST(cumsum_i32, SIZE({16, 4, 2, 3}), 3, torch::kInt32)

TEST_F(HpuOpTest, cumsum_without_dtype_attr) {
  torch::ScalarType dtype = torch::kInt32;
  GenerateInputs(1, {{3, 4, 8, 16}}, dtype);
  auto expected = torch::cumsum(GetCpuInput(0), -2);
  auto result = torch::cumsum(GetHpuInput(0), -2);
  Compare(expected, result);
}

TEST_F(HpuOpTest, cumsum_without_dtype_attr_2) {
  torch::ScalarType dtype = torch::kFloat;
  GenerateInputs(1, {{8, 24, 24, 3}}, dtype);
  auto expected = torch::cumsum(GetCpuInput(0), 2);
  auto result = torch::cumsum(GetHpuInput(0), 2);
  Compare(expected, result);
}

TEST_F(HpuOpTest, cumsum_without_dtype_attr_3) {
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, {{8, 24, 24}}, dtype);
  auto expected = torch::cumsum(GetCpuInput(0), 0);
  auto result = torch::cumsum(GetHpuInput(0), 0);
  Compare(expected, result);
}
