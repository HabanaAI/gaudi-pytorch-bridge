/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
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
