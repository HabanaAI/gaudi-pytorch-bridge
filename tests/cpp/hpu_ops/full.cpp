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
#define SHAPE(...) __VA_ARGS__

class HpuOpTest : public HpuOpTestUtil {
  c10::ScalarType old_dtype_ = c10::ScalarType::Undefined;
  void SetUp() override {
    old_dtype_ = torch::get_default_dtype_as_scalartype();
    DisableCpuFallback();
    TearDownBridge();
  }
  void TearDown() override {
    torch::set_default_dtype(c10::scalarTypeToTypeMeta(old_dtype_));
    RestoreMode();
  }
};

#define DEF_FULL_OP_TEST(name, shape, dtype, default_dtype)        \
  TEST_F(HpuOpTest, name) {                                        \
    auto fillValue = GenerateScalar<int>(-128, 127);               \
    if (default_dtype) {                                           \
      torch::set_default_dtype(c10::scalarTypeToTypeMeta(dtype));  \
    }                                                              \
    auto hpuResult = at::native::full(                             \
        shape,                                                     \
        fillValue,                                                 \
        default_dtype ? c10::nullopt : c10::make_optional(dtype),  \
        c10::nullopt,                                              \
        c10::Device(c10::DeviceType::HPU));                        \
    auto cpuResult = at::native::full(                             \
        shape,                                                     \
        fillValue,                                                 \
        default_dtype ? c10::nullopt : c10::make_optional(dtype)); \
    Compare(cpuResult, hpuResult, 0, 0);                           \
  }

DEF_FULL_OP_TEST(full_1x_f32_false, SHAPE({{1}}), torch::kFloat32, false)
DEF_FULL_OP_TEST(
    full_2x3x4x_f32_false,
    SHAPE({{2, 3, 4}}),
    torch::kFloat32,
    false)
DEF_FULL_OP_TEST(
    full_2x3x4x_f32_true,
    SHAPE({{2, 3, 4}}),
    torch::kFloat32,
    true)
DEF_FULL_OP_TEST(
    full_2x3x4x_bf16_false,
    SHAPE({{2, 3, 4}}),
    torch::kBFloat16,
    false)
DEF_FULL_OP_TEST(
    full_2x3x4x_bf16_true,
    SHAPE({{2, 3, 4}}),
    torch::kBFloat16,
    true)
DEF_FULL_OP_TEST(full_2x3x4x_i8_false, SHAPE({{2, 3, 4}}), torch::kChar, false)
DEF_FULL_OP_TEST(full_2x3x4x_i8_true, SHAPE({{2, 3, 4}}), torch::kChar, true)
DEF_FULL_OP_TEST(full_2x3x4x_i32_false, SHAPE({{2, 3, 4}}), torch::kInt, false)
DEF_FULL_OP_TEST(full_2x3x4x_i32_true, SHAPE({{2, 3, 4}}), torch::kInt, true)
DEF_FULL_OP_TEST(
    full_2x3x4x_i64_false,
    SHAPE({{2, 3, 4}}),
    torch::kInt64,
    false)
DEF_FULL_OP_TEST(full_2x3x4x_i64_true, SHAPE({{2, 3, 4}}), torch::kInt64, true)
