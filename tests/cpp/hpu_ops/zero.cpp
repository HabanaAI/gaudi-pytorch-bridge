/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "backend/habana_device/HPUGuardImpl.h"
#include "util.h"

#define TORCH_TYPE(type) torch::k##type
#define SHAPE(...) __VA_ARGS__

#define ZEROS_TEST(type, shape)                                         \
  TEST_F(HpuOpTest, zeros_##type) {                                     \
    torch::ScalarType dtype = TORCH_TYPE(type);                         \
    auto expected = torch::zeros(shape, dtype);                         \
    auto result =                                                       \
        torch::zeros(shape, torch::TensorOptions(dtype).device("hpu")); \
    Compare(expected, result);                                          \
  }

class HpuOpTest : public HpuOpTestUtil {};

ZEROS_TEST(BFloat16, SHAPE({{1}}))
ZEROS_TEST(Float, SHAPE({{1, 3, 5}}))
ZEROS_TEST(Int, SHAPE({{}}))
ZEROS_TEST(Char, SHAPE({{1, 3, 5, 7, 9}}))
ZEROS_TEST(Byte, SHAPE({{3}}))
ZEROS_TEST(Short, SHAPE({{2, 4}}))
ZEROS_TEST(Half, SHAPE({{1, 3, 5, 7, 9}}))
ZEROS_TEST(Long, SHAPE({{4, 5}}))
