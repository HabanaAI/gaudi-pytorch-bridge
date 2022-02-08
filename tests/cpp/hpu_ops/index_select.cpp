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

#define INDEX_SELECT_OUT_TEST(                                             \
    test_name, in_size, max_value, datatype, index_value, dim, out_size)   \
  TEST_F(HpuOpTest, test_name) {                                           \
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
class HpuOpTest : public HpuOpTestUtil {};
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
INDEX_SELECT_OUT_TEST(
    index_select_out_3D,
    SIZE({8, 512, 512}),
    512,
    torch::kFloat,
    4,
    2,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_4D,
    SIZE({8, 24, 24, 3}),
    24,
    torch::kChar,
    3,
    2,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_5D,
    SIZE({8, 24, 24, 16, 36}),
    36,
    torch::kBFloat16,
    8,
    4,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_int,
    SIZE({512, 256, 400}),
    400,
    torch::kInt,
    6,
    2,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_char,
    SIZE({200, 356, 20}),
    200,
    torch::kChar,
    4,
    0,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_neg_float,
    SIZE({8, 24, 24, 16, 36}),
    36,
    torch::kFloat,
    6,
    -1,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_neg_bfloat,
    SIZE({8, 24, 24, 16, 36}),
    24,
    torch::kBFloat16,
    2,
    -4,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_neg_char,
    SIZE({8, 24, 24, 16, 36}),
    24,
    torch::kChar,
    2,
    -1,
    0)

INDEX_SELECT_OUT_TEST(
    index_select_out_shape,
    SIZE({8, 24, 32, 3}),
    31,
    torch::kChar,
    2,
    2,
    32)