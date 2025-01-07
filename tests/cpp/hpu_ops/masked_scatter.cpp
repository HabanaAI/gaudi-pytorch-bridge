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

class MaskedScatterHpuOpTest : public HpuOpTestUtil {
 public:
  void generateAllInputsAndCompare(
      torch::ArrayRef<torch::IntArrayRef> tensorsShapes,
      torch::ArrayRef<torch::ScalarType> tensorsTypes) {
    GenerateInputs(3, std::move(tensorsShapes), std::move(tensorsTypes));
    auto expected =
        torch::masked_scatter(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
    auto result =
        torch::masked_scatter(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));
    Compare(expected, result);
  }

  void generateSelfSourceApplyMaskAndCompare(
      torch::ArrayRef<torch::IntArrayRef> tensorsShapes,
      torch::ArrayRef<torch::ScalarType> tensorsTypes,
      const torch::Tensor& mask) {
    GenerateInputs(2, tensorsShapes, tensorsTypes);

    auto expected = torch::masked_scatter(GetCpuInput(0), mask, GetCpuInput(1));
    auto result =
        torch::masked_scatter(GetHpuInput(0), mask.to("hpu"), GetHpuInput(1));
    Compare(expected, result);
  }
};

TEST_F(MaskedScatterHpuOpTest, AllTensorsWithRankOne) {
  generateAllInputsAndCompare(
      {{8}, {1}, {8}}, {at::kFloat, at::kBool, at::kFloat});
}

TEST_F(
    MaskedScatterHpuOpTest,
    SelfAndMaskHaveTheSameShapeSourceIsFlattenedAlready) {
  generateAllInputsAndCompare(
      {{3, 5}, {3, 5}, {15}}, {at::kFloat, at::kBool, at::kFloat});
}

TEST_F(
    MaskedScatterHpuOpTest,
    MasksFirstDimHaveToBeBroadcastedSourceIsFlattenedAlready) {
  generateAllInputsAndCompare(
      {{3, 5}, {5}, {15}}, {at::kFloat, at::kBool, at::kFloat});
}

TEST_F(
    MaskedScatterHpuOpTest,
    MasksSecondDimHaveToBeBroadcastedSourceIsFlattenedAlready) {
  generateAllInputsAndCompare(
      {{3, 5}, {3, 1}, {15}}, {at::kFloat, at::kBool, at::kFloat});
}

TEST_F(
    MaskedScatterHpuOpTest,
    MasksSecondFourthFifthDimHaveToBeBroadcastedSourceIsFlattenedAlready) {
  generateAllInputsAndCompare(
      {{3, 5, 4, 7, 2}, {3, 1, 4, 1, 1}, {840}},
      {at::kFloat, at::kBool, at::kFloat});
}

TEST_F(MaskedScatterHpuOpTest, SourceNeedsToBeFlattened) {
  generateAllInputsAndCompare(
      {{3, 5}, {3, 5}, {5, 3}}, {at::kFloat, at::kBool, at::kFloat});
}

TEST_F(MaskedScatterHpuOpTest, MaskNeedsToBeBroadcastedAndSourceFlattened) {
  generateAllInputsAndCompare(
      {{2, 2, 4, 2, 2}, {1, 1, 1, 1, 1}, {2, 2, 4, 2, 2}},
      {at::kChar, at::kBool, at::kChar});
}

TEST_F(MaskedScatterHpuOpTest, MaskWithAllZeros) {
  generateSelfSourceApplyMaskAndCompare(
      {{16, 32}, {4, 4, 32}},
      {torch::kBFloat16, torch::kBFloat16},
      torch::zeros({1, 1}).to(torch::kBool));
}

TEST_F(MaskedScatterHpuOpTest, MaskWithAllOnes) {
  generateSelfSourceApplyMaskAndCompare(
      {{8, 8, 8}, {512}},
      {torch::kInt32, torch::kInt32},
      torch::ones({8, 1, 8}).to(torch::kBool));
}

TEST_F(MaskedScatterHpuOpTest, SelfAndSourceSizesDiffer) {
  generateSelfSourceApplyMaskAndCompare(
      {{2, 4, 5}, {2, 4, 1}},
      {torch::kFloat, torch::kFloat},
      torch::tensor({0, 1, 0, 0, 0}).to(torch::kBool));
}
