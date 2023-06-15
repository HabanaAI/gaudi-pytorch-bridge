/*******************************************************************************
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

class BatchNormHpuOpTest : public HpuOpTestUtil {
  template <typename... Ts, std::size_t... Is>
  void compare_output_impl(
      const std::tuple<Ts...>& expected,
      const std::tuple<Ts...>& result,
      std::index_sequence<Is...>) {
    (Compare(std::get<Is>(expected), std::get<Is>(result)), ...);
  }

 public:
  template <typename... Ts>
  void CompareOutputs(
      const std::tuple<Ts...>& expected,
      const std::tuple<Ts...>& result) {
    compare_output_impl(expected, result, std::index_sequence_for<Ts...>{});
  }
};

class NativeBatchNormLegitNoStatsHpuOpTest
    : public BatchNormHpuOpTest,
      public ::testing::WithParamInterface<std::tuple<
          std::vector<long int>,
          torch::ScalarType,
          bool,
          float,
          float>> {
 public:
  void generateAndCompare(
      const torch::IntArrayRef sizes,
      const torch::ScalarType dtype,
      const bool training,
      const float momentum,
      const float epsilon) {
    GenerateInputs(3, {sizes, {sizes[1]}, {sizes[1]}}, {dtype});

    auto expected = torch::_native_batch_norm_legit(
        GetCpuInput(0),
        GetCpuInput(1),
        GetCpuInput(2),
        training,
        momentum,
        epsilon);
    auto result = torch::_native_batch_norm_legit(
        GetHpuInput(0),
        GetHpuInput(1),
        GetHpuInput(2),
        training,
        momentum,
        epsilon);

    CompareOutputs(expected, result);
  }
};

TEST_P(NativeBatchNormLegitNoStatsHpuOpTest, nativeBatchNormLegitNoStatsTest) {
  const auto [inputTensorShape, inputTensorType, training, momentum, epsilon] =
      GetParam();

  if (not training)
    GTEST_SKIP(); // TODO [Jira: SW-150573] seg fault visible for CPU pytorch
                  // result

  generateAndCompare(
      inputTensorShape, inputTensorType, training, momentum, epsilon);
}

INSTANTIATE_TEST_CASE_P(
    BatchNormTests,
    NativeBatchNormLegitNoStatsHpuOpTest,
    ::testing::Values(
        std::make_tuple(
            std::vector<long int>{2, 3, 4, 5},
            at::kFloat,
            true,
            0.999f,
            1e-5f),
        std::make_tuple(
            std::vector<long int>{5, 4, 3, 2},
            at::kFloat,
            false,
            0.999f,
            1e-5f)));
