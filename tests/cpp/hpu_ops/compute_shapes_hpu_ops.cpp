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
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum);
  auto result = torch::binary_cross_entropy(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ GetHpuInput(2),
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
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum,
      expected);
  result = torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ GetHpuInput(2),
      at::Reduction::Sum,
      result);

  Compare(expected, result);
}

#define SIZE(...) __VA_ARGS__

#define INDEX_SELECT_OUT_TEST(                                             \
    test_name, in_size, max_value, datatype, index_value, dim, out_size)   \
  TEST_F(HpuOpComputeShapeTest, test_name) {                               \
    torch::ScalarType dtype = datatype;                                    \
    GenerateIntInputs(1, {index_value}, 0, max_value);                     \
    auto cpu_index = GetCpuInput(0).to(torch::kLong);                      \
    auto hpu_index = cpu_index.to(torch::kHPU);                            \
    auto expected = torch::empty(out_size, dtype);                         \
    auto result =                                                          \
        torch::empty(out_size, torch::TensorOptions(dtype).device("hpu")); \
    GenerateInputs(1, {in_size}, {dtype});                                 \
    torch::index_select_outf(GetCpuInput(0), dim, cpu_index, expected);    \
    torch::index_select_outf(GetHpuInput(0), dim, hpu_index, result);      \
    Compare(expected, result, 0, 0);                                       \
  }

#define INDEX_SELECT_TEST(                                               \
    test_name, in_size, max_value, datatype, index_value, dim, out_size) \
  TEST_F(HpuOpComputeShapeTest, test_name) {                             \
    torch::ScalarType dtype = datatype;                                  \
    GenerateIntInputs(1, {index_value}, 0, max_value);                   \
    auto cpu_index = GetCpuInput(0).to(torch::kLong);                    \
    auto hpu_index = cpu_index.to(torch::kHPU);                          \
    GenerateInputs(1, {in_size}, {dtype});                               \
    auto expected = torch::index_select(GetCpuInput(0), dim, cpu_index); \
    auto result = torch::index_select(GetHpuInput(0), dim, hpu_index);   \
    Compare(expected, result, 0, 0);                                     \
  }

class HpuOpTest : public HpuOpTestUtil {};
INDEX_SELECT_TEST(index_select_1D, SIZE({1024}), 1024, torch::kInt, 4, 0, 0)
INDEX_SELECT_OUT_TEST(index_select_2D, SIZE({28, 28}), 28, torch::kInt, 5, 1, 0)

INDEX_SELECT_OUT_TEST(
    index_select_out_1D,
    SIZE({1024}),
    1024,
    torch::kInt,
    4,
    0,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_2D,
    SIZE({28, 28}),
    28,
    torch::kInt,
    5,
    1,
    0)
