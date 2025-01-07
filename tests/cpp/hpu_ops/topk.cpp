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

#include "../utils/device_type_util.h"
#include "util.h"

#define HPU_TOPK_OUT_TEST(name, op_code, dtype, sorted, largest)              \
  TEST_F(HpuOpTest, name) {                                                   \
    /* Test sporadically failing on Gaudi3: SW-162282 */                      \
    if (isGaudi3()) {                                                         \
      GTEST_SKIP() << "Test skipped on Gaudi3.";                              \
    }                                                                         \
    GenerateInputs(1, {{8, 24, 24, 3}}, dtype);                               \
    auto k = 5;                                                               \
    auto dim = 0;                                                             \
    auto result = torch::empty(0).to(dtype);                                  \
    auto result_h = result.to("hpu");                                         \
    auto indices = torch::empty(0).to(torch::kLong);                          \
    auto indices_h = indices.to("hpu");                                       \
    torch::op_code(GetCpuInput(0), k, dim, largest, sorted, result, indices); \
    torch::op_code(                                                           \
        GetHpuInput(0), k, dim, largest, sorted, result_h, indices_h);        \
    Compare(result, result_h);                                                \
  }

#define HPU_TOPK_TEST(name, op_code, dtype, sorted, largest)                 \
  TEST_F(HpuOpTest, name) {                                                  \
    /* Test sporadically failing on Gaudi3: SW-160563 & SW-160568 */         \
    if (isGaudi3()) {                                                        \
      GTEST_SKIP() << "Test skipped on Gaudi3.";                             \
    }                                                                        \
    GenerateInputs(1, {{8, 24, 24, 3}}, dtype);                              \
    auto k = 5;                                                              \
    auto dim = 0;                                                            \
    auto expected = torch::op_code(GetCpuInput(0), k, dim, largest, sorted); \
    auto result = torch::op_code(GetHpuInput(0), k, dim, largest, sorted);   \
    Compare(std::get<0>(expected), std::get<0>(result));                     \
  }
class HpuOpTest : public HpuOpTestUtil {};

HPU_TOPK_OUT_TEST(topk_out_Float32, topk_outf, torch::kFloat32, true, false)
HPU_TOPK_OUT_TEST(topk_out_BFloat16, topk_outf, torch::kBFloat16, true, true)
HPU_TOPK_OUT_TEST(topk_out_Int32, topk_outf, torch::kInt32, true, true)
HPU_TOPK_OUT_TEST(topk_out_Int16, topk_outf, torch::kInt16, true, true)

HPU_TOPK_TEST(topk_Float32, topk, torch::kFloat32, true, false)
HPU_TOPK_TEST(topk_BFloat16, topk, torch::kBFloat16, true, true)
HPU_TOPK_TEST(topk_Int32, topk, torch::kInt32, true, false)
HPU_TOPK_TEST(topk_Int16, topk, torch::kInt16, true, true)

TEST_F(HpuOpTest, topk) {
  GenerateInputs(1, {{10, 3, 2}});
  auto k = 3;
  auto dim = 0;
  bool sorted = true;
  bool largest = true;

  auto expected = torch::topk(GetCpuInput(0), k, dim, largest, sorted);
  auto result = torch::topk(GetHpuInput(0), k, dim, largest, sorted);

  Compare(std::get<0>(expected), std::get<0>(result));
  Compare(std::get<1>(expected), std::get<1>(result));
}

TEST_F(HpuOpTest, topk_Float16) {
  if (isGaudi()) {
    GTEST_SKIP() << "Test skipped on Gaudi.";
  }
  GenerateInputs(1, {{10, 3, 2}});
  auto k = 3;
  auto dim = 0;
  bool sorted = true;
  bool largest = true;

  auto expected = torch::topk(GetCpuInput(0), k, dim, largest, sorted);
  auto result =
      torch::topk(GetHpuInput(0).to(torch::kFloat16), k, dim, largest, sorted);

  Compare(std::get<0>(expected), std::get<0>(result).to(torch::kFloat));
  Compare(std::get<1>(expected), std::get<1>(result));
}

TEST_F(HpuOpTest, topk_out) {
  GenerateInputs(1, {{8, 24, 24, 3}});
  auto k = 5;
  auto dim = 0;
  bool sorted = true;
  bool largest = true;
  auto result = torch::empty(0);
  auto result_h = result.to("hpu");
  auto indices = torch::empty(0).to(torch::kLong);
  auto indices_h = indices.to("hpu");

  torch::topk_outf(GetCpuInput(0), k, dim, largest, sorted, result, indices);
  torch::topk_outf(
      GetHpuInput(0), k, dim, largest, sorted, result_h, indices_h);

  Compare(result, result_h);
  Compare(indices, indices_h);
}