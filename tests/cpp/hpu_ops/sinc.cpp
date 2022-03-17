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

constexpr double bf16_rtol = 0.32;

const std::vector<float> input_tensor{1.0, 0.0, 3.0};

// sinc out variant
#define HPU_SINC_OUT_TEST(name, in_size)        \
  TEST_F(HpuOpTest, name) {                     \
    GenerateInputs(1, {in_size});               \
    auto expected = torch::empty(0);            \
    auto result = torch::empty(0, "hpu");       \
    torch::sinc_outf(GetCpuInput(0), expected); \
    torch::sinc_outf(GetHpuInput(0), result);   \
    Compare(expected, result);                  \
  }

// sinc usual variant
#define HPU_SINC_USUAL_TEST(name, in_size)       \
  TEST_F(HpuOpTest, name) {                      \
    GenerateInputs(1, {in_size});                \
    auto expected = torch::sinc(GetCpuInput(0)); \
    auto result = torch::sinc(GetHpuInput(0));   \
    Compare(expected, result);                   \
  }

// sinc inplace variant
#define HPU_SINC_INPLACE_TEST(name, in_size) \
  TEST_F(HpuOpTest, name) {                  \
    GenerateInputs(1, {in_size});            \
    torch::sinc_(GetCpuInput(0));            \
    torch::sinc_(GetHpuInput(0));            \
    Compare(GetCpuInput(0), GetHpuInput(0)); \
  }

#define SINC_TEST(name, in_size)               \
  HPU_SINC_OUT_TEST(name##_out, SIZE(in_size)) \
  HPU_SINC_USUAL_TEST(name, SIZE(in_size))     \
  HPU_SINC_INPLACE_TEST(name##_, SIZE(in_size))

class HpuOpTest : public HpuOpTestUtil {};

SINC_TEST(sinc_1dim, SIZE({1024}))

SINC_TEST(sinc_2dim, SIZE({2, 16}))

SINC_TEST(sinc_3dim, SIZE({2, 3, 4}))

SINC_TEST(sinc_4dim, SIZE({2, 4, 4, 8}))

SINC_TEST(sinc_5dim, SIZE({4, 2, 4, 8, 8}))

// sinc usual variant with 0 included in the test case
TEST_F(HpuOpTest, sinc_4dim_rand) {
  GenerateInputs(1, {{12, 16, 24, 0}});

  auto expected = torch::sinc(GetCpuInput(0));
  auto result = torch::sinc(GetHpuInput(0));

  Compare(expected, result);
}

// sinc out variant for float datatype
TEST_F(HpuOpTest, sinc_out) {
  GenerateInputs(1, {{2, 3}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sinc_outf(GetCpuInput(0), expected);
  torch::sinc_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

// sinc inplace variant
TEST_F(HpuOpTest, sinc_) {
  GenerateInputs(1);

  torch::sinc_(GetCpuInput(0));
  torch::sinc_(GetHpuInput(0));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

/*
 * Below test will fail for BFloat16 for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-69264
 */
// sinc inplace variant for bf16 datatype
TEST_F(HpuOpTest, sinc_bf16) {
  GenerateInputs(1, torch::kBFloat16);

  torch::sinc_(GetCpuInput(0));
  torch::sinc_(GetHpuInput(0));

  Compare(GetCpuInput(0), GetHpuInput(0), bf16_rtol);
}

// input x=[0.0]
TEST_F(HpuOpTest, sinc_zero) {
  auto tensor1 = torch::tensor(input_tensor);
  auto tensor2 = torch::tensor(input_tensor, "hpu");
  auto expected = torch::sinc(tensor1);
  auto result = torch::sinc(tensor2);
  Compare(expected, result);
}

// 0dim input tensor
TEST_F(HpuOpTest, sinc_0dim) {
  auto tensor1 = torch::tensor(45.0);
  auto tensor2 = torch::tensor(45.0, "hpu");
  auto expected = torch::sinc(tensor1);
  auto result = torch::sinc(tensor2);
  Compare(expected, result);
}

// input x=inf and -inf
TEST_F(HpuOpTest, sinc_inf) {
  float f_pos_inf = std::numeric_limits<float>::infinity();
  float f_neg_inf = -std::numeric_limits<float>::infinity();
  auto tensor1 = torch::tensor({f_pos_inf, f_neg_inf});
  auto tensor2 = torch::tensor({f_pos_inf, f_neg_inf}, "hpu");
  auto expected = torch::sinc(tensor1);
  auto result = torch::sinc(tensor2);

  Compare(expected, result);
}