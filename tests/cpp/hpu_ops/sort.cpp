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

#define HPU_SORT_OUT_TEST(name, op_code, dtype, descending)                   \
  TEST_F(HpuOpTest, name) {                                                   \
    GenerateInputs(1, {{2, 3, 10}}, dtype);                                   \
    int dim = 2;                                                              \
    c10::optional<bool> stable(false);                                        \
    auto result = torch::empty(0).to(dtype);                                  \
    auto result_h = result.to("hpu");                                         \
    auto indices = torch::empty(0).to(torch::kLong);                          \
    auto indices_h = indices.to("hpu");                                       \
    torch::op_code(GetCpuInput(0), stable, dim, descending, result, indices); \
    torch::op_code(                                                           \
        GetHpuInput(0), stable, dim, descending, result_h, indices_h);        \
    Compare(result, result_h);                                                \
  }

#define HPU_SORT_TEST(name, op_code, dtype, descending)              \
  TEST_F(HpuOpTest, name) {                                          \
    GenerateInputs(1, {{2, 3, 10}}, dtype);                          \
    int dim = 2;                                                     \
    auto expected = torch::op_code(GetCpuInput(0), dim, descending); \
    auto result = torch::op_code(GetHpuInput(0), dim, descending);   \
    Compare(std::get<0>(expected), std::get<0>(result));             \
    Compare(std::get<1>(expected), std::get<1>(result));             \
  }
class HpuOpTest : public HpuOpTestUtil {};

HPU_SORT_OUT_TEST(sort_out_Float32_asc, sort_outf, torch::kFloat32, false)
HPU_SORT_OUT_TEST(sort_out_BFloat16_asc, sort_outf, torch::kBFloat16, false)
HPU_SORT_OUT_TEST(sort_out_Int32_asc, sort_outf, torch::kInt32, false)
HPU_SORT_OUT_TEST(sort_out_Int16_asc, sort_outf, torch::kInt16, false)

HPU_SORT_OUT_TEST(sort_out_Float32_desc, sort_outf, torch::kFloat32, true)
HPU_SORT_OUT_TEST(sort_out_BFloat16_desc, sort_outf, torch::kBFloat16, true)
HPU_SORT_OUT_TEST(sort_out_Int32_desc, sort_outf, torch::kInt32, true)
HPU_SORT_OUT_TEST(sort_out_Int16_desc, sort_outf, torch::kInt16, true)

HPU_SORT_TEST(sort_Float32_desc, sort, torch::kFloat32, true)
HPU_SORT_TEST(sort_BFloat16_desc, sort, torch::kBFloat16, true)
HPU_SORT_TEST(sort_Int32_asc, sort, torch::kInt32, false)
HPU_SORT_TEST(sort_Int16_asc, sort, torch::kInt16, false)

TEST_F(HpuOpTest, sort_Float16_asc) {
  if (isGaudi()) {
    GTEST_SKIP() << "Test skipped on Gaudi.";
  }
  GenerateInputs(1, {{10, 3, 2}});
  auto k = 3;
  auto dim = 0;
  bool sorted = true;
  bool largest = false;

  auto expected = torch::sort(GetCpuInput(0), dim, largest);
  auto result = torch::sort(GetHpuInput(0).to(torch::kFloat16), dim, largest);

  Compare(std::get<0>(expected), std::get<0>(result).to(torch::kFloat));
  Compare(std::get<1>(expected), std::get<1>(result));
}

TEST_F(HpuOpTest, sort) {
  GenerateInputs(1, {{10, 3, 2}});
  auto k = 3;
  auto dim = 0;
  bool sorted = true;
  bool largest = true;

  auto expected = torch::sort(GetCpuInput(0), dim, largest);
  auto result = torch::sort(GetHpuInput(0), dim, largest);

  Compare(std::get<0>(expected), std::get<0>(result));
  Compare(std::get<1>(expected), std::get<1>(result));
}

TEST_F(HpuOpTest, sort_out) {
  GenerateInputs(1, {{8, 24, 24, 3}});
  int dim = 2;
  c10::optional<bool> stable(false);
  bool descending = false;
  auto result = torch::empty(0);
  auto result_h = result.to("hpu");
  auto indices = torch::empty(0).to(torch::kLong);
  auto indices_h = indices.to("hpu");

  torch::sort_outf(GetCpuInput(0), stable, dim, descending, result, indices);
  torch::sort_outf(
      GetHpuInput(0), stable, dim, descending, result_h, indices_h);
  Compare(result, result_h);
  Compare(indices, indices_h);
}
