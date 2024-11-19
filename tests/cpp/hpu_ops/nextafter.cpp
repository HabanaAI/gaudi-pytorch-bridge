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

class HpuOpTest : public HpuOpTestUtil {
 public:
  void testNextafter(
      torch::ArrayRef<torch::IntArrayRef> sizes,
      torch::ScalarType dtype) {
    GenerateInputs(2, sizes, dtype);

    auto expected = torch::nextafter(GetCpuInput(0), GetCpuInput(1));
    auto result = torch::nextafter(GetHpuInput(0), GetHpuInput(1));

    // nextafter returns next floating-point value so need to let no tolerance
    Compare(expected, result, 0.0, 0.0);
  }

  void testNextafterOut(
      torch::ArrayRef<torch::IntArrayRef> sizes,
      torch::ScalarType dtype) {
    GenerateInputs(2, sizes, {dtype, dtype});
    auto expected = torch::empty(0, dtype);
    auto result = torch::empty(0, dtype).to(torch::kHPU);

    torch::nextafter_outf(GetCpuInput(0), GetCpuInput(1), expected);
    torch::nextafter_outf(GetHpuInput(0), GetHpuInput(1), result);

    Compare(expected, result, 0.0, 0.0);
  }

  void testNextafter_(
      torch::ArrayRef<torch::IntArrayRef> sizes,
      torch::ScalarType dtype) {
    GenerateInputs(2, sizes, dtype);
    auto input = GetCpuInput(0);
    auto hinput = GetHpuInput(0);

    input.nextafter_(GetCpuInput(1));
    hinput.nextafter_(GetHpuInput(1));

    Compare(input, hinput, 0.0, 0.0);
  }
};

TEST_F(HpuOpTest, nextafterOut) {
  testNextafterOut({{4, 3, 1}, {4, 3, 1}}, torch::kFloat);
  testNextafterOut({{4, 3, 1}, {4, 3, 1}}, torch::kBFloat16);
}

TEST_F(HpuOpTest, nextafterOutBroadcast) {
  testNextafterOut({{4, 3, 1, 5}, {4, 3, 1, 1}}, torch::kFloat);
  testNextafterOut({{4, 3, 1, 1}, {4, 3, 1, 3}}, torch::kFloat);
  testNextafterOut({{2, 3, 4}, {3, 4}}, torch::kFloat);

  testNextafterOut({{4, 3, 1, 5}, {4, 3, 1, 1}}, torch::kBFloat16);
  testNextafterOut({{4, 3, 1, 1}, {4, 3, 1, 3}}, torch::kBFloat16);
  testNextafterOut({{2, 3, 4}, {3, 4}}, torch::kBFloat16);
}

TEST_F(HpuOpTest, nextafterBroadcast) {
  testNextafter({{4, 3, 1}, {4, 3, 1}}, torch::kFloat);
  testNextafter({{2, 3, 4}, {3, 4}}, torch::kFloat);
  testNextafter({{4, 3, 3}, {4, 3, 1}}, torch::kFloat);

  testNextafter({{4, 3, 1}, {4, 3, 1}}, torch::kBFloat16);
  testNextafter({{2, 3, 4}, {3, 4}}, torch::kBFloat16);
  testNextafter({{4, 3, 3}, {4, 3, 1}}, torch::kBFloat16);
}

TEST_F(HpuOpTest, nextafter_) {
  testNextafter_({{4, 3, 1}, {4, 3, 1}}, torch::kFloat);
  testNextafter_({{4, 3, 1}, {4, 3, 1}}, torch::kBFloat16);
}

TEST_F(HpuOpTest, nextafter_Broadcast) {
  testNextafter_({{4, 3, 1}, {4, 3, 1}}, torch::kFloat);
  testNextafter_({{4, 3, 3}, {4, 3, 1}}, torch::kFloat);

  testNextafter_({{4, 3, 1}, {4, 3, 1}}, torch::kBFloat16);
  testNextafter_({{4, 3, 3}, {4, 3, 1}}, torch::kBFloat16);
}
