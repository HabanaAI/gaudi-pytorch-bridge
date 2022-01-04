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

#define HPU_MEDIAN_TEST(name, in_size, dtype)       \
  TEST_F(HpuOpTest, name) {                         \
    auto input = torch::randn({in_size}).to(dtype); \
    auto hinput = input.to(torch::kHPU);            \
    auto expected = torch::median(input);           \
    auto result = torch::median(hinput);            \
    Compare(expected, result);                      \
  }

#define HPU_MEDIAN_DIM_TEST(name, in_size, axis, keepdim, dtype) \
  TEST_F(HpuOpTest, name) {                                      \
    auto input = torch::randn({in_size}).to(dtype);              \
    auto hinput = input.to(torch::kHPU);                         \
    auto expected = torch::median(input, axis, keepdim);         \
    auto result = torch::median(hinput, axis, keepdim);          \
    Compare(std::get<0>(expected), std::get<0>(result));         \
    Compare(std::get<1>(expected), std::get<1>(result));         \
  }

class HpuOpTest : public HpuOpTestUtil {};

HPU_MEDIAN_TEST(median_float, SIZE({8, 24, 24, 24, 5}), torch::kFloat)
HPU_MEDIAN_TEST(median_bfloat, SIZE({8, 5, 4, 8}), torch::kBFloat16)

HPU_MEDIAN_DIM_TEST(
    mediandim_float,
    SIZE({8, 24, 24, 5}),
    3,
    true,
    torch::kFloat)

HPU_MEDIAN_DIM_TEST(
    mediandim_float_negative,
    SIZE({12, 8, 12, 8}),
    -2,
    true,
    torch::kFloat)

HPU_MEDIAN_DIM_TEST(
    mediandim_float_false,
    SIZE({4, 8, 4}),
    0,
    false,
    torch::kFloat)

HPU_MEDIAN_DIM_TEST(
    mediandim_bfloat,
    SIZE({8, 8, 16}),
    2,
    true,
    torch::kBFloat16)

HPU_MEDIAN_DIM_TEST(
    mediandim_bfloat_negative,
    SIZE({12, 12}),
    0,
    true,
    torch::kBFloat16)

HPU_MEDIAN_DIM_TEST(
    mediandim_bfloat_false,
    SIZE({24, 24}),
    1,
    false,
    torch::kBFloat16)