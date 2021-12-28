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

#define HPU_FLOOR_DIVIDE_INPLACE_TEST(                       \
    name, in_size1, in_size2, dtype1, dtype2)                \
  TEST_F(HpuOpTest, name) {                                  \
    auto self = torch::randn({in_size1}).to(dtype1);         \
    auto other = torch::randint(1, 100, {in_size2}, dtype2); \
    auto self_h = self.to(torch::kHPU);                      \
    auto other_h = other.to(torch::kHPU);                    \
    self.floor_divide_(other);                               \
    self_h.floor_divide_(other_h);                           \
    Compare(self, self_h);                                   \
  }

#define HPU_FLOOR_DIVIDE_OUT_TEST(name, in_size1, in_size2, dtype1, dtype2)    \
  TEST_F(HpuOpTest, name) {                                                    \
    torch::manual_seed(0);                                                     \
    auto self = torch::randn({in_size1}).to(dtype1);                           \
    auto other = torch::randint(1, 100, {in_size2}, dtype2);                   \
    auto self_h = self.to(torch::kHPU);                                        \
    auto other_h = other.to(torch::kHPU);                                      \
    auto expected = torch::empty(0, dtype1);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype1).device("hpu")); \
    torch::floor_divide_outf(self, other, expected);                           \
    torch::floor_divide_outf(self_h, other_h, result);                         \
    Compare(expected, result);                                                 \
  }

class HpuOpTest : public HpuOpTestUtil {};

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_float,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kFloat)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_float_broadcast,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 1, 1}),
    torch::kFloat,
    torch::kFloat)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_bfloat_float,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kBFloat16,
    torch::kFloat)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_bfloat_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kBFloat16,
    torch::kInt)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_float_bfloat,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kBFloat16)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_float_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kInt)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_int_int,
    SIZE({8, 4, 4, 4}),
    SIZE({8, 4, 4, 4}),
    torch::kInt,
    torch::kInt)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_float_float,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kFloat)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_float_bfloat,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kBFloat16)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_float_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kInt)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_bfloat_bfloat,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 1, 1}),
    torch::kBFloat16,
    torch::kBFloat16)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_bfloat_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kBFloat16,
    torch::kInt)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_bfloat_float,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kBFloat16,
    torch::kFloat)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_int_int,
    SIZE({8, 4, 4, 4}),
    SIZE({8, 4, 4, 4}),
    torch::kInt,
    torch::kInt)