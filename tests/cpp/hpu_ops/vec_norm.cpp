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

#define HPU_VEC_NORM(name, in_size, dtype, dim, keepdim, ord, out_type) \
  TEST_F(HpuOpTest, name) {                                             \
    GenerateInputs(1, {in_size}, {dtype});                              \
    std::vector<int64_t> dims = dim;                                    \
    auto expected = torch::linalg_vector_norm(                          \
        GetCpuInput(0), ord, dims, keepdim, out_type);                  \
    auto result = torch::linalg_vector_norm(                            \
        GetHpuInput(0), ord, dims, keepdim, out_type);                  \
    Compare(expected, result);                                          \
  }

#define HPU_VEC_NORM_OUT(name, in_size, dtype, dim, keepdim, ord, out_type)   \
  TEST_F(HpuOpTest, name) {                                                   \
    GenerateInputs(1, {in_size}, {dtype});                                    \
    std::vector<int64_t> dims = dim;                                          \
    auto exp = torch::empty(0, out_type);                                     \
    auto res = torch::empty(0, torch::TensorOptions(out_type).device("hpu")); \
    torch::linalg_vector_norm_outf(                                           \
        GetCpuInput(0), ord, dims, keepdim, out_type, exp);                   \
    torch::linalg_vector_norm_outf(                                           \
        GetHpuInput(0), ord, dims, keepdim, out_type, res);                   \
    Compare(exp, res);                                                        \
  }

class HpuOpTest : public HpuOpTestUtil {};

HPU_VEC_NORM(
    lvn_f32,
    SIZE({2, 3}),
    torch::kFloat,
    SIZE({1}),
    true,
    2,
    torch::kFloat)

HPU_VEC_NORM(
    lvn_base_f32,
    SIZE({2, 3}),
    torch::kFloat,
    SIZE({1}),
    true,
    -2,
    torch::kFloat)

HPU_VEC_NORM(
    lvn_base_bf16,
    SIZE({2, 3}),
    torch::kBFloat16,
    SIZE({1}),
    false,
    -2,
    torch::kBFloat16)

HPU_VEC_NORM(
    lvn3d,
    SIZE({2, 5, 4}),
    torch::kFloat,
    SIZE({0, 1}),
    false,
    1,
    torch::kFloat)

HPU_VEC_NORM(
    lvn1d,
    SIZE({5}),
    torch::kBFloat16,
    SIZE({0}),
    true,
    0,
    torch::kFloat)

HPU_VEC_NORM(
    lvninf,
    SIZE({2, 3, 4, 5, 6}),
    torch::kFloat,
    SIZE({0, 2}),
    true,
    INFINITY,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvn_kfal,
    SIZE({2, 3, 4, 5}),
    torch::kFloat,
    SIZE({}),
    false,
    0.5,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvninfn,
    SIZE({2, 3, 4, 5, 6}),
    torch::kFloat,
    SIZE({0, 2}),
    false,
    -INFINITY,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvn_3ord,
    SIZE({2, 3, 4}),
    torch::kFloat,
    SIZE({1, 2}),
    false,
    3,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvn_5d,
    SIZE({2, 3, 4, 5, 6}),
    torch::kFloat,
    SIZE({3}),
    false,
    -3,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvn6,
    SIZE({2, 3}),
    torch::kBFloat16,
    SIZE({1}),
    false,
    -2.0,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvn7,
    SIZE({3, 2, 4}),
    torch::kBFloat16,
    SIZE({1}),
    false,
    -2.0,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvn8,
    SIZE({2, 3}),
    torch::kBFloat16,
    SIZE({1}),
    true,
    0.33,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvn_ord2_f32,
    SIZE({2, 3}),
    torch::kFloat,
    SIZE({1}),
    false,
    2.0,
    torch::kFloat)

HPU_VEC_NORM_OUT(
    lvn_ord2_bf16,
    SIZE({3, 2, 4}),
    torch::kBFloat16,
    SIZE({1}),
    false,
    2.0,
    torch::kFloat)

/*
 * Below test will fail for BFloat16 for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-94726
 */
TEST_F(HpuOpTest, lvn_dim) {
  auto ord = 2;
  bool keepdim = false;
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, {{102}}, {torch::kBFloat16});

  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::linalg_vector_norm_outf(
      GetCpuInput(0), ord, {} /*dim*/, keepdim, dtype, exp);
  torch::linalg_vector_norm_outf(
      GetHpuInput(0), ord, {} /*dim*/, keepdim, dtype, res);
  Compare(exp, res, 0.017, 0.02);
}
