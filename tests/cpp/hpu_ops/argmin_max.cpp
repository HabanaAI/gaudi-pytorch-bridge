/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "util.h"

#define SIZE(...) __VA_ARGS__

#define HPU_ARG_MIN_MAX_OUT_TEST(                                              \
    name, op_code, in_size, out_size, dim, keepdim, dtype)                     \
  TEST_F(ArgminmaxHpuOpTest, name) {                                           \
    auto out_dtype = torch::kLong;                                             \
    auto input = torch::randn({in_size}).to(dtype);                            \
    auto hinput = input.to(torch::kHPU);                                       \
    auto expected = torch::empty(out_size, out_dtype);                         \
    auto result =                                                              \
        torch::empty(out_size, torch::TensorOptions(out_dtype).device("hpu")); \
    torch::op_code(input, dim, keepdim, expected);                             \
    torch::op_code(hinput, dim, keepdim, result);                              \
    Compare(expected, result, 0, 0);                                           \
  }

#define HPU_ARG_MIN_MAX_USUAL_TEST(name, op, in_size, dim, keepdim, dtype) \
  TEST_F(ArgminmaxHpuOpTest, name) {                                       \
    auto input = torch::randn(in_size).to(dtype);                          \
    auto hinput = input.to(torch::kHPU);                                   \
    auto expected = torch::op(input, dim, keepdim);                        \
    auto result = torch::op(hinput, dim, keepdim);                         \
    Compare(expected, result, 0, 0);                                       \
  }

// "argmax_cpu" not implemented for 'Bool'
#define HPU_ARG_MIN_MAX_USUAL_BOOL_TEST(                                     \
    name, op, in_size, dim, keepdim, dtype)                                  \
  TEST_F(ArgminmaxHpuOpTest, name) {                                         \
    GenerateInputs(1, {in_size}, dtype);                                     \
    auto expected = torch::op(GetCpuInput(0).to(torch::kInt), dim, keepdim); \
    auto result = torch::op(GetHpuInput(0), dim, keepdim);                   \
    Compare(expected, result, 0, 0);                                         \
  }

class ArgminmaxHpuOpTest : public HpuOpTestUtil {};

HPU_ARG_MIN_MAX_OUT_TEST(
    argmax_4d_out_keepdim,
    argmax_outf,
    SIZE({3, 4, 5, 6}),
    SIZE({3, 1, 5, 6}),
    1,
    true,
    torch::kBFloat16)
HPU_ARG_MIN_MAX_OUT_TEST(
    argmax_3d_out_reduce_dim,
    argmax_outf,
    SIZE({4, 5, 6}),
    SIZE({4, 5}),
    -1,
    false,
    torch::kFloat)
HPU_ARG_MIN_MAX_OUT_TEST(
    argmax_4d_out_reduce_dim_global_max,
    argmax_outf,
    SIZE({2, 3, 4, 5}),
    SIZE({0}),
    SIZE({}),
    false,
    torch::kFloat)
HPU_ARG_MIN_MAX_OUT_TEST(
    argmax_3d_out_keepdim_global_max,
    argmax_outf,
    SIZE({3, 8, 5}),
    SIZE({0}),
    SIZE({}),
    true,
    torch::kFloat)
HPU_ARG_MIN_MAX_OUT_TEST(
    argmax_4d_out_reduce_dim_global_max_int,
    argmax_outf,
    SIZE({2, 3, 4, 5}),
    SIZE({0}),
    SIZE({}),
    false,
    torch::kInt)
HPU_ARG_MIN_MAX_OUT_TEST(
    argmax_3d_out_keepdim_global_max_int,
    argmax_outf,
    SIZE({2, 3, 4}),
    SIZE({2, 3}),
    -1,
    false,
    torch::kInt)
HPU_ARG_MIN_MAX_OUT_TEST(
    argmin_4d_out_keepdim,
    argmin_outf,
    SIZE({4, 5, 3, 6}),
    SIZE({4, 5, 1, 6}),
    2,
    true,
    torch::kFloat)
HPU_ARG_MIN_MAX_OUT_TEST(
    argmin_3d_out_reduce_dim,
    argmin_outf,
    SIZE({2, 3, 4}),
    SIZE({2, 3}),
    -1,
    false,
    torch::kBFloat16)
HPU_ARG_MIN_MAX_OUT_TEST(
    argmin_3d_out_reduce_dim_global_min,
    argmin_outf,
    SIZE({3, 4, 5}),
    SIZE({0}),
    SIZE({}),
    false,
    torch::kBFloat16)
HPU_ARG_MIN_MAX_USUAL_TEST(
    argmin_4d_reduce_dim,
    argmin,
    SIZE({8, 4, 6, 3}),
    -3,
    false,
    torch::kFloat)
HPU_ARG_MIN_MAX_USUAL_TEST(
    argmin_4d_reduce_dim_global_min,
    argmin,
    SIZE({5, 4, 8, 2}),
    SIZE({}),
    false,
    torch::kBFloat16)
HPU_ARG_MIN_MAX_USUAL_TEST(
    argmin_3d_keepdim,
    argmin,
    SIZE({4, 7, 3}),
    0,
    true,
    torch::kFloat)
HPU_ARG_MIN_MAX_USUAL_TEST(
    argmin_3d_keepdim_global_min,
    argmin,
    SIZE({8, 2, 5}),
    SIZE({}),
    true,
    torch::kBFloat16)

HPU_ARG_MIN_MAX_USUAL_TEST(
    argmax_4d_reduce_dim,
    argmax,
    SIZE({8, 4, 6, 3}),
    -3,
    false,
    torch::kFloat)
HPU_ARG_MIN_MAX_USUAL_TEST(
    argmax_4d_reduce_dim_global_min,
    argmax,
    SIZE({5, 4, 8, 2}),
    SIZE({2}),
    false,
    torch::kBFloat16)
HPU_ARG_MIN_MAX_USUAL_TEST(
    argmax_3d_keepdim,
    argmax,
    SIZE({4, 7, 3}),
    SIZE({}),
    true,
    torch::kFloat)
HPU_ARG_MIN_MAX_USUAL_TEST(
    argmax_3d_keepdim_global_min,
    argmax,
    SIZE({8, 2, 5}),
    -3,
    true,
    torch::kInt)

HPU_ARG_MIN_MAX_USUAL_BOOL_TEST(
    argmax_bool_true,
    argmax,
    SIZE({8, 2, 5}),
    SIZE({1}),
    true,
    torch::kBool)
HPU_ARG_MIN_MAX_USUAL_BOOL_TEST(
    argmax_bool_false,
    argmax,
    SIZE({8, 2, 5}),
    -2,
    false,
    torch::kBool)
