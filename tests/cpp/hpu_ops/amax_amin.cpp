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
#define DIM(...) __VA_ARGS__

#define HPU_AMAX_AMIN_OUT_TEST(name, op_code, in_size, dim, keepdim, dtype)   \
  TEST_F(HpuOpTest, name) {                                                   \
    GenerateInputs(1, {in_size}, dtype);                                      \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op_code(GetCpuInput(0), dim, keepdim, expected);                   \
    torch::op_code(GetHpuInput(0), dim, keepdim, result);                     \
    Compare(expected, result);                                                \
  }

#define HPU_AMAX_AMIN_USUAL_TEST(name, op_code, in_size, dim, keepdim, dtype) \
  TEST_F(HpuOpTest, name) {                                                   \
    GenerateInputs(1, {in_size}, dtype);                                      \
    auto expected = torch::op_code(GetCpuInput(0), dim, keepdim);             \
    auto result = torch::op_code(GetHpuInput(0), dim, keepdim);               \
    Compare(expected, result);                                                \
  }

#define HPU_AMINMAX_USUAL_TEST(name, op_code, in_size, dim, keepdim, dtype) \
  TEST_F(HpuOpTest, name) {                                                 \
    GenerateInputs(1, {in_size}, dtype);                                    \
    auto exp = torch::op_code(GetCpuInput(0), dim, keepdim);                \
    auto res = torch::op_code(GetHpuInput(0), dim, keepdim);                \
    Compare(std::get<0>(exp), std::get<0>(res), 0, 0);                      \
    Compare(std::get<1>(exp), std::get<1>(res), 0, 0);                      \
  }

// aminmax cpu not support bf16 but reduce_min/max supported
#define HPU_AMINMAX_USUAL_BF16_TEST(name, op_code, in_size, dim, keepdim)      \
  TEST_F(HpuOpTest, name) {                                                    \
    torch::ScalarType dtype = torch::kBFloat16;                                \
    GenerateInputs(1, {in_size}, dtype);                                       \
    auto exp = torch::op_code(GetCpuInput(0).to(torch::kFloat), dim, keepdim); \
    auto res = torch::op_code(GetHpuInput(0), dim, keepdim);                   \
    Compare(std::get<0>(exp).to(dtype), std::get<0>(res), 0, 0);               \
    Compare(std::get<1>(exp).to(dtype), std::get<1>(res), 0, 0);               \
  }

#define HPU_AMINMAX_OUT_TEST(name, op_code, in_size, dim, keepdim, dtype)   \
  TEST_F(HpuOpTest, name) {                                                 \
    GenerateInputs(1, {in_size}, dtype);                                    \
    auto min = torch::empty(0, dtype);                                      \
    auto hmin = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    auto max = torch::empty(0, dtype);                                      \
    auto hmax = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op_code(GetCpuInput(0), dim, keepdim, min, max);                 \
    torch::op_code(GetHpuInput(0), dim, keepdim, hmin, hmax);               \
    Compare(min, hmin, 0, 0);                                               \
    Compare(max, hmax, 0, 0);                                               \
  }

// aminmax_out cpu not support bf16 but reduce_min/max supported
#define HPU_AMINMAX_OUT_BF16_TEST(name, op_code, in_size, dim, keepdim)       \
  TEST_F(HpuOpTest, name) {                                                   \
    torch::ScalarType dtype = torch::kBFloat16;                               \
    GenerateInputs(1, {in_size}, dtype);                                      \
    auto min = torch::empty(0, torch::kFloat);                                \
    auto hmin = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));   \
    auto max = torch::empty(0, torch::kFloat);                                \
    auto hmax = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));   \
    torch::op_code(GetCpuInput(0).to(torch::kFloat), dim, keepdim, min, max); \
    torch::op_code(GetHpuInput(0), dim, keepdim, hmin, hmax);                 \
    Compare(min.to(dtype), hmin, 0, 0);                                       \
    Compare(max.to(dtype), hmax, 0, 0);                                       \
  }

class HpuOpTest : public HpuOpTestUtil {};

HPU_AMAX_AMIN_OUT_TEST(
    amax_long,
    amax_outf,
    SIZE({2, 3}),
    DIM({0, 1}),
    true,
    torch::kInt64)
HPU_AMAX_AMIN_OUT_TEST(
    amax_double,
    amax_outf,
    SIZE({3, 32, 32}),
    DIM({0, -2}),
    false,
    torch::kFloat64)
HPU_AMAX_AMIN_OUT_TEST(
    amax_float,
    amax_outf,
    SIZE({1, 2, 2, 5}),
    2,
    false,
    torch::kFloat32)
HPU_AMAX_AMIN_USUAL_TEST(
    amax_bfloat,
    amax,
    SIZE({1, 32, 32}),
    -2,
    true,
    torch::kBFloat16)
HPU_AMAX_AMIN_USUAL_TEST(
    amax_int,
    amax,
    SIZE({2, 2, 3, 32, 32}),
    DIM({0, 1, -3}),
    true,
    torch::kInt32)
HPU_AMAX_AMIN_OUT_TEST(
    amin_long,
    amin_outf,
    SIZE({2, 3}),
    DIM({0, 1}),
    true,
    torch::kInt64)
HPU_AMAX_AMIN_OUT_TEST(
    amin_double,
    amin_outf,
    SIZE({3, 32, 32}),
    DIM({0, -2}),
    false,
    torch::kFloat64)
HPU_AMAX_AMIN_OUT_TEST(
    amin_float,
    amin_outf,
    SIZE({1, 2, 2, 5}),
    2,
    false,
    torch::kFloat32)
HPU_AMAX_AMIN_USUAL_TEST(
    amin_bfloat,
    amin,
    SIZE({1, 32, 32}),
    -2,
    true,
    torch::kBFloat16)
HPU_AMAX_AMIN_USUAL_TEST(
    amin_int,
    amin,
    SIZE({2, 2, 3, 32, 32}),
    DIM({0, 1, -3}),
    true,
    torch::kInt32)

HPU_AMINMAX_USUAL_TEST(
    aminmax_float,
    aminmax,
    SIZE({1, 32, 32}),
    -2,
    true,
    torch::kFloat)
HPU_AMINMAX_USUAL_TEST(
    aminmax_double,
    aminmax,
    SIZE({2, 3, 4, 5}),
    -2,
    true,
    torch::kFloat64)
HPU_AMINMAX_USUAL_TEST(
    aminmax_long,
    aminmax,
    SIZE({3, 6, 5}),
    1,
    false,
    torch::kInt64)
HPU_AMINMAX_USUAL_TEST(
    aminmax_dim_none,
    aminmax,
    SIZE({1, 2, 4, 32}),
    c10::nullopt,
    false,
    torch::kFloat)
HPU_AMINMAX_USUAL_TEST(
    aminmax_int,
    aminmax,
    SIZE({2, 2, 3, 32, 32}),
    2,
    false,
    torch::kInt32)
HPU_AMINMAX_USUAL_TEST(
    aminmax_bool_false,
    aminmax,
    SIZE({2, 3, 6}),
    2,
    false,
    torch::kBool)
HPU_AMINMAX_USUAL_TEST(
    aminmax_bool_true,
    aminmax,
    SIZE({2, 3, 5, 6}),
    -1,
    true,
    torch::kBool)

HPU_AMINMAX_USUAL_BF16_TEST(
    aminmax_bf16_false,
    aminmax,
    SIZE({3, 4, 5, 6}),
    3,
    false)
HPU_AMINMAX_USUAL_BF16_TEST(
    aminmax_bf16_true,
    aminmax,
    SIZE({2, 3, 6}),
    -2,
    true)

HPU_AMINMAX_OUT_TEST(
    aminmax_out_float,
    aminmax_outf,
    SIZE({1, 32, 32}),
    0,
    false,
    torch::kFloat)
HPU_AMINMAX_OUT_TEST(
    aminmax_out_dim_none,
    aminmax_outf,
    SIZE({1, 32, 32}),
    c10::nullopt,
    false,
    torch::kFloat)
HPU_AMINMAX_OUT_TEST(
    aminmax_out_int,
    aminmax_outf,
    SIZE({2, 2, 3, 32, 32}),
    0,
    true,
    torch::kInt32)
HPU_AMINMAX_OUT_TEST(
    aminmax_out_double,
    aminmax_outf,
    SIZE({2, 4, 5, 8}),
    -1,
    false,
    torch::kFloat64)
HPU_AMINMAX_OUT_TEST(
    aminmax_out_long,
    aminmax_outf,
    SIZE({3, 4, 5, 6}),
    1,
    true,
    torch::kInt64)
HPU_AMINMAX_OUT_TEST(
    aminmax_out_bool_false,
    aminmax_outf,
    SIZE({2, 3, 4, 6}),
    -3,
    false,
    torch::kBool)
HPU_AMINMAX_OUT_TEST(
    aminmax_out_bool_true,
    aminmax_outf,
    SIZE({2, 4, 6}),
    1,
    true,
    torch::kBool)

HPU_AMINMAX_OUT_BF16_TEST(
    aminmax_out_bf16_false,
    aminmax_outf,
    SIZE({3, 4, 5, 6}),
    3,
    false)
HPU_AMINMAX_OUT_BF16_TEST(
    aminmax_out_bf16_true,
    aminmax_outf,
    SIZE({2, 3, 6}),
    -2,
    true)