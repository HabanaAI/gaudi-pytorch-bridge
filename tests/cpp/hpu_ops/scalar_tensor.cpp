/******************************************************************************
 * Copyright (C) 2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

#define DEF_DTYPE(datatype) .dtype(datatype)
#define DEF_NO_DTYPE(datatype)

#define DEF_SCALAR_TENSOR_BASIC_TEST(name, dtype_defined, datatype, randtype)  \
  TEST_F(HpuOpTest, name) {                                                    \
    GenerateInputs(1);                                                         \
    auto scalar = c10::Scalar(GenerateScalar<randtype>());                     \
    auto expected = torch::scalar_tensor(                                      \
        scalar, torch::TensorOptions().device("cpu") dtype_defined(datatype)); \
    auto result = torch::scalar_tensor(                                        \
        scalar, torch::TensorOptions().device("hpu") dtype_defined(datatype)); \
    Compare(expected, result);                                                 \
  }

DEF_SCALAR_TENSOR_BASIC_TEST(scalar_tensor_int, DEF_DTYPE, torch::kInt, int)
DEF_SCALAR_TENSOR_BASIC_TEST(scalar_tensor_char, DEF_DTYPE, torch::kChar, int)
DEF_SCALAR_TENSOR_BASIC_TEST(scalar_tensor_long, DEF_DTYPE, torch::kLong, int)
DEF_SCALAR_TENSOR_BASIC_TEST(
    scalar_tensor_float,
    DEF_DTYPE,
    torch::kFloat,
    float)
DEF_SCALAR_TENSOR_BASIC_TEST(
    scalar_tensor_kbfloat16,
    DEF_DTYPE,
    torch::kBFloat16,
    float)
DEF_SCALAR_TENSOR_BASIC_TEST(scalar_tensor_int_no_dtype, DEF_NO_DTYPE, 0, int)
DEF_SCALAR_TENSOR_BASIC_TEST(
    scalar_tensor_float_no_dtype,
    DEF_NO_DTYPE,
    0,
    float)
