/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest-param-test.h>
#include "habana_kernels/fallback_helper.h"
#include "util.h"

#define TENSOR_TYPE_float torch::kFloat32
#define TENSOR_TYPE_bfloat16 torch::kBFloat16
#define TENSOR_TYPE_int64 torch::kInt64
#define TENSOR_TYPE_int32 torch::kInt32
#define TENSOR_TYPE_int16 torch::kInt16
#define TENSOR_TYPE_int8 torch::kInt8
#define TENSOR_TYPE_uint8 torch::kUInt8
#define TENSOR_TYPE_bool torch::kBool
#define TENSOR_TYPE_byte torch::kByte

#define GET_TENSOR_TYPE(type) TENSOR_TYPE_##type

#define Gather2DOutTest(test_name, dtype)                                    \
  TEST_F(HpuOpTest, test_name) {                                             \
    GenerateInputs(1, {{2, 3}}, GET_TENSOR_TYPE(dtype));                     \
    auto cpu_index = torch::tensor({{0, 1, 1}, {1, 0, 1}}).to(torch::kLong); \
    auto hpu_index = cpu_index.to(torch::kHPU);                              \
    auto expected = torch::empty(0).to(GET_TENSOR_TYPE(dtype));              \
    auto result = expected.to(torch::kHPU);                                  \
    torch::gather_outf(                                                      \
        GetCpuInput(0), 0, cpu_index, /*sparse_grad*/ false, expected);      \
    torch::gather_outf(                                                      \
        GetHpuInput(0), 0, hpu_index, /*sparse_grad*/ false, result);        \
    Compare(GetCpuInput(0), GetHpuInput(0));                                 \
  }

#define Gather3DOutTest(test_name, dtype)                                \
  TEST_F(HpuOpTest, test_name) {                                         \
    GenerateInputs(1, {{2, 3, 4}}, GET_TENSOR_TYPE(dtype));              \
    auto cpu_index =                                                     \
        torch::tensor({{{0, 0}, {1, 0}, {1, 1}}}).to(torch::kLong);      \
    auto hpu_index = cpu_index.to(torch::kHPU);                          \
    auto expected = torch::empty(0).to(GET_TENSOR_TYPE(dtype));          \
    auto result = expected.to(torch::kHPU);                              \
    torch::gather_outf(                                                  \
        GetCpuInput(0), -1, cpu_index, /*sparse_grad*/ false, expected); \
    torch::gather_outf(                                                  \
        GetHpuInput(0), -1, hpu_index, /*sparse_grad*/ false, result);   \
    Compare(GetCpuInput(0), GetHpuInput(0));                             \
  }

#define Gather4DOutTest(test_name, dtype)                                      \
  TEST_F(HpuOpTest, test_name) {                                               \
    GenerateInputs(1, {{4, 5, 6, 7}}, GET_TENSOR_TYPE(dtype));                 \
    auto cpu_index =                                                           \
        torch::tensor({{{{0, 1, 2}, {1, 2, 0}, {1, 3, 0}}}}).to(torch::kLong); \
    auto hpu_index = cpu_index.to(torch::kHPU);                                \
    auto expected = torch::empty(0).to(GET_TENSOR_TYPE(dtype));                \
    auto result = expected.to(torch::kHPU);                                    \
    torch::gather_outf(                                                        \
        GetCpuInput(0), 0, cpu_index, /*sparse_grad*/ false, expected);        \
    torch::gather_outf(                                                        \
        GetHpuInput(0), 0, hpu_index, /*sparse_grad*/ false, result);          \
    Compare(GetCpuInput(0), GetHpuInput(0));                                   \
  }

#define Gather5DOutTest(test_name, dtype)                                   \
  TEST_F(HpuOpTest, test_name) {                                            \
    GenerateInputs(1, {{4, 5, 6, 7, 8}}, GET_TENSOR_TYPE(dtype));           \
    auto cpu_index = torch::tensor({{{{{0, 1, 2}, {1, 2, 0}, {1, 3, 0}}}}}) \
                         .to(torch::kLong);                                 \
    auto hpu_index = cpu_index.to(torch::kHPU);                             \
    auto expected = torch::empty(0).to(GET_TENSOR_TYPE(dtype));             \
    auto result = expected.to(torch::kHPU);                                 \
    torch::gather_outf(                                                     \
        GetCpuInput(0), 0, cpu_index, /*sparse_grad*/ false, expected);     \
    torch::gather_outf(                                                     \
        GetHpuInput(0), 0, hpu_index, /*sparse_grad*/ false, result);       \
    Compare(GetCpuInput(0), GetHpuInput(0));                                \
  }

class HpuOpTest : public HpuOpTestUtil {};

Gather2DOutTest(gather_out_int8, int8);
Gather2DOutTest(gather_out_uint8, uint8);
Gather3DOutTest(gather_out_int16, int16);
Gather3DOutTest(gather_out_bfloat16, bfloat16);
Gather4DOutTest(gather_out_float, float);
Gather4DOutTest(gather_out_byte, byte);
Gather5DOutTest(gather_out_int32, int32);
Gather5DOutTest(gather_out_bool, bool);

class GatherDtypeSupportTest : public DTypeSupportTest<c10::ScalarType> {};

TEST_P(GatherDtypeSupportTest, GatherOutDtypeSupportTest) {
  auto dtype = GetParam();
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kHPU);
  auto input = torch::tensor({1, 2}, options);
  auto output = torch::empty(2, options);
  auto index = torch::tensor({0, 0}, options.dtype(torch::kInt64));

  auto result =
      torch::gather_outf(input, 0, index, false, output).to(torch::kCPU);
  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(
      op_fallback_frequency.find("aten::gather.out"),
      op_fallback_frequency.end());
}

INSTANTIATE_TEST_SUITE_P(
    TypeSupportTest,
    GatherDtypeSupportTest,
    testing::Values(
        torch::kBFloat16,
        torch::kFloat32,
        torch::kInt32,
        torch::kInt8,
        torch::kInt16));
