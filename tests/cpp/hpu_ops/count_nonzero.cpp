//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
// All Rights Reserved.
//
// Unauthorized copying of this file or any element(s) within it, via any medium
// is strictly prohibited.
// This file contains Habana Labs, Ltd. proprietary and confidential information
// and is subject to the confidentiality and license agreements under which it
// was provided.
//
//===----------------------------------------------------------------------===//

#include "../utils/dtype_supported_on_device.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {
 private:
  std::tuple<at::Tensor, at::Tensor> prepare_input(
      torch::IntArrayRef inputSize,
      torch::ScalarType dtype) {
    GenerateInputs(1, {inputSize}, dtype);
    auto mask = torch::randint(2, inputSize);
    auto hpu_input = torch::mul(GetHpuInput(0), mask.to(torch::kHPU));
    auto cpu_input = torch::mul(GetCpuInput(0), mask);
    return {hpu_input, cpu_input};
  }

 public:
  void testCountNonZero(
      torch::IntArrayRef inputSize,
      torch::ScalarType dtype,
      at::optional<at::IntArrayRef> dims,
      at::optional<int64_t> dim) {
    auto [hpu_input, cpu_input] = prepare_input(inputSize, dtype);
    at::Tensor hpu_result, cpu_result;

    if (dims.has_value()) {
      hpu_result = torch::count_nonzero(hpu_input, dims.value());
      cpu_result = torch::count_nonzero(cpu_input, dims.value());
    } else {
      hpu_result = torch::count_nonzero(hpu_input, dim);
      cpu_result = torch::count_nonzero(cpu_input, dim);
    }

    Compare(cpu_result, hpu_result);
  }

  void testCountNonZeroOut(
      torch::IntArrayRef inputSize,
      torch::ScalarType dtype,
      at::optional<at::IntArrayRef> dims,
      at::optional<int64_t> dim) {
    auto [hpu_input, cpu_input] = prepare_input(inputSize, dtype);
    auto cpu_result = torch::empty({0}, torch::kLong);
    auto hpu_result = cpu_result.to(torch::kHPU);

    if (dims.has_value()) {
      torch::count_nonzero_out(hpu_result, hpu_input, dims.value());
      torch::count_nonzero_out(cpu_result, cpu_input, dims.value());
    } else {
      torch::count_nonzero_out(hpu_result, hpu_input, dim);
      torch::count_nonzero_out(cpu_result, cpu_input, dim);
    }

    Compare(cpu_result, hpu_result);
  }
};

#define COUNT_NON_ZERO_TEST(DTYPE)                                          \
  TEST_F(HpuOpTest, count_nonzero_##DTYPE) {                                \
    if (!IsDtypeSupportedOnCurrentDevice(torch::DTYPE)) {                   \
      GTEST_SKIP();                                                         \
    }                                                                       \
    testCountNonZero({3, 2, 4}, torch::DTYPE, at::IntArrayRef{0, 1, 2}, 0); \
    testCountNonZero(                                                       \
        {3, 2, 4, 6, 2, 1}, torch::DTYPE, at::IntArrayRef{3, 1, 2}, 0);     \
    testCountNonZero({3, 2, 4, 3, 3}, torch::DTYPE, at::nullopt, 2);        \
    testCountNonZero({2, 3, 4, 5}, torch::DTYPE, at::IntArrayRef{}, 0);     \
    testCountNonZero({2, 3, 4, 5}, torch::DTYPE, at::nullopt, at::nullopt); \
  }

#define COUNT_NON_ZERO_OUT_TEST(DTYPE)                                         \
  TEST_F(HpuOpTest, count_nonzero_out_##DTYPE) {                               \
    if (!IsDtypeSupportedOnCurrentDevice(torch::DTYPE)) {                      \
      GTEST_SKIP();                                                            \
    }                                                                          \
    testCountNonZeroOut({3, 2, 4}, torch::DTYPE, at::IntArrayRef{0, 1, 2}, 0); \
    testCountNonZeroOut(                                                       \
        {3, 2, 4, 6, 2, 1}, torch::DTYPE, at::IntArrayRef{3, 1, 2}, 0);        \
    testCountNonZeroOut({3, 2, 4, 3, 3}, torch::DTYPE, at::nullopt, 2);        \
    testCountNonZeroOut({2, 3, 4, 5}, torch::DTYPE, at::IntArrayRef{}, 0);     \
    testCountNonZeroOut({3, 2, 4, 3}, torch::DTYPE, at::nullopt, at::nullopt); \
  }

#define COUNT_NON_ZERO_TESTS(DTYPE) \
  COUNT_NON_ZERO_TEST(DTYPE) COUNT_NON_ZERO_OUT_TEST(DTYPE)

COUNT_NON_ZERO_TESTS(kFloat32);
COUNT_NON_ZERO_TESTS(kBFloat16);
COUNT_NON_ZERO_TESTS(kFloat16);
COUNT_NON_ZERO_TESTS(kInt32);
COUNT_NON_ZERO_TESTS(kInt16);
COUNT_NON_ZERO_TESTS(kInt8);
