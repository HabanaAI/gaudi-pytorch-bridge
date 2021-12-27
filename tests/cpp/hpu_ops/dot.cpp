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

#define TENSOR_TYPE_float torch::kFloat
#define TENSOR_TYPE_bfloat16 torch::kBFloat16

#define GET_TENSOR_TYPE(type) TENSOR_TYPE_##type

#define HPU_DOT_OUT_TEST(type)                                                \
  TEST_F(HpuOpTest, dot_out_##type) {                                         \
    torch::ScalarType dtype = GET_TENSOR_TYPE(type);                          \
    GenerateInputs(2, {{10}, {10}}, dtype);                                   \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::dot_outf(GetCpuInput(0), GetCpuInput(1), expected);                \
    torch::dot_outf(GetHpuInput(0), GetHpuInput(1), result);                  \
    Compare(expected, result);                                                \
  }
class HpuOpTest : public HpuOpTestUtil {};
HPU_DOT_OUT_TEST(float);
HPU_DOT_OUT_TEST(bfloat16);
