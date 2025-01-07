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

#define HPU_VIEW_SIZE_TEST(name, in_size, DTYPE, out_size) \
  TEST_F(HpuOpTest, name) {                                \
    GenerateInputs(1, {in_size}, {DTYPE});                 \
    auto expected = GetCpuInput(0).view(out_size);         \
    auto result = GetHpuInput(0).view(out_size);           \
    Compare(expected, result);                             \
  }

class HpuOpTest : public HpuOpTestUtil {};

/**
 * Test cases fail for view op
 * Issue Raised: https://jira.habana-labs.com/browse/SW-95463
 */

HPU_VIEW_SIZE_TEST(view_size, SIZE({1024}), torch::kInt64, SIZE({2, 512}))
