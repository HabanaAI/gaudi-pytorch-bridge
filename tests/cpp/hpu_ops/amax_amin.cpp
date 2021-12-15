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
  TEST_F(AmaxAminHpuOpTest, name) {                                           \
    auto input = torch::randn({in_size}).to(dtype);                           \
    auto hinput = input.to(torch::kHPU);                                      \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op_code(input, dim, keepdim, expected);                            \
    torch::op_code(hinput, dim, keepdim, result);                             \
    Compare(expected, result);                                                \
  }

#define HPU_AMAX_AMIN_USUAL_TEST(name, op_code, in_size, dim, keepdim, dtype) \
  TEST_F(AmaxAminHpuOpTest, name) {                                           \
    auto input = torch::randn(in_size).to(dtype);                             \
    auto hinput = input.to(torch::kHPU);                                      \
    auto expected = torch::op_code(input, dim, keepdim);                      \
    auto result = torch::op_code(hinput, dim, keepdim);                       \
    Compare(expected, result);                                                \
  }

class AmaxAminHpuOpTest : public HpuOpTestUtil {};

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