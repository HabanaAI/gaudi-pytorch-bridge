/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "../utils/device_type_util.h"
#include "util.h"
#define SIZE(...) __VA_ARGS__
#define DTYPE torch::kFloat

#define HPU_MEDIAN_TEST(name, in_size)             \
  TEST_F(HpuOpTest, name) {                        \
    GenerateInputs(1, {{in_size}}, {DTYPE});       \
    auto expected = torch::median(GetCpuInput(0)); \
    auto result = torch::median(GetHpuInput(0));   \
    Compare(expected, result);                     \
  }

#define HPU_MEDIAN_DIM_TEST(name, in_size, axis, keepdim)         \
  TEST_F(HpuOpTest, name) {                                       \
    /* Test sporadically failing on Gaudi3: SW-165423 */          \
    if (isGaudi3()) {                                             \
      GTEST_SKIP() << "Test skipped on Gaudi3.";                  \
    }                                                             \
    GenerateInputs(1, {{in_size}}, {DTYPE});                      \
    auto expected = torch::median(GetCpuInput(0), axis, keepdim); \
    auto result = torch::median(GetHpuInput(0), axis, keepdim);   \
    Compare(std::get<0>(expected), std::get<0>(result));          \
  }

#define HPU_MEDIAN_DIM_VALUES_TEST(                                         \
    name, in_size, expected_size, axis, keepdim)                            \
  TEST_F(HpuOpTest, name) {                                                 \
    /* Test sporadically failing on Gaudi3: SW-167683 */                    \
    if (isGaudi3()) {                                                       \
      GTEST_SKIP() << "Test skipped on Gaudi3.";                            \
    }                                                                       \
    GenerateInputs(1, {{in_size}}, {DTYPE});                                \
    auto expected_value = torch::empty({expected_size}, DTYPE);             \
    auto expected_index = torch::empty({expected_size}, torch::kLong);      \
    auto result_value = torch::empty(                                       \
        {expected_size}, torch::TensorOptions(DTYPE).device("hpu"));        \
    auto result_index = torch::empty(                                       \
        {expected_size}, torch::TensorOptions(torch::kLong).device("hpu")); \
    torch::median_outf(                                                     \
        GetCpuInput(0), axis, keepdim, expected_value, expected_index);     \
    torch::median_outf(                                                     \
        GetHpuInput(0), axis, keepdim, result_value, result_index);         \
    Compare(expected_value, result_value);                                  \
  }

/**
 * Test cases fail when the size of the tensor in the reduction axis >37
 * Issue Raised: https://jira.habana-labs.com/browse/SW-70095
 * Mis-match in the Median index value
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68143
 */

class HpuOpTest : public HpuOpTestUtil {};

HPU_MEDIAN_TEST(median_float_1, SIZE({24}))
HPU_MEDIAN_TEST(median_float_2, SIZE({17}))
HPU_MEDIAN_TEST(median_float_3, SIZE({6, 6}))
HPU_MEDIAN_TEST(median_float_4, SIZE({5, 7}))
HPU_MEDIAN_TEST(median_float_5, SIZE({2, 5, 3}))
HPU_MEDIAN_TEST(median_float_6, SIZE({1, 1, 1, 1}))
HPU_MEDIAN_TEST(median_float_7, SIZE({2, 3, 4, 1}))
HPU_MEDIAN_TEST(median_float_8, SIZE({2, 2, 2, 2, 2}))

HPU_MEDIAN_DIM_TEST(mediandim_float, SIZE({8, 24, 24, 5}), 3, true)

HPU_MEDIAN_DIM_TEST(mediandim_float_negative, SIZE({12, 8, 12, 8}), -2, true)

HPU_MEDIAN_DIM_TEST(mediandim_float_false, SIZE({64, 24, 4}), 1, false)

HPU_MEDIAN_DIM_VALUES_TEST(
    mediandimvalue_float,
    SIZE({12, 4, 8, 16}),
    SIZE({12, 4, 8, 1}),
    3,
    true)

HPU_MEDIAN_DIM_VALUES_TEST(
    mediandimvalue_float_false,
    SIZE({24, 8, 16}),
    SIZE({8, 16}),
    0,
    false)

HPU_MEDIAN_DIM_VALUES_TEST(
    mediandimvalue_float_negative,
    SIZE({16, 8, 16}),
    SIZE({16, 8, 1}),
    -1,
    true)
