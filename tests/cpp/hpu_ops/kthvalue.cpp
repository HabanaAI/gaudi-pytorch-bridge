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

#include "../utils/device_type_util.h"
#include "../utils/dtype_supported_on_device.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {
 private:
  void compareKthvalueResults(
      at::Tensor& resultValues,
      at::Tensor& resultIndices,
      at::Tensor& expectedValues,
      at::Tensor& self,
      int axis,
      int keep_dims) {
    Compare(expectedValues, resultValues);

    if (!keep_dims) {
      resultIndices = torch::unsqueeze(resultIndices, axis);
      expectedValues = torch::unsqueeze(expectedValues, axis);
    }
    auto valuesByIndices =
        torch::gather(self, axis, resultIndices.to(torch::kCPU))
            .to(torch::kHPU);
    Compare(expectedValues, valuesByIndices);
  }

 public:
  void testKthvalue(
      torch::IntArrayRef inputSize,
      torch::ScalarType dtype,
      int k_value,
      int axis,
      bool keep_dims) {
    GenerateInputs(1, {inputSize}, dtype);

    auto [resultValues, resultIndices] =
        torch::kthvalue(GetHpuInput(0), k_value, axis, keep_dims);

    auto self = GetCpuInput(0);
    if (dtype == torch::kFloat16) {
      self = self.to(torch::kFloat);
    }
    auto [expectedValues, expectedIndices] =
        torch::kthvalue(self, k_value, axis, keep_dims);
    if (dtype == torch::kFloat16) {
      expectedValues = expectedValues.to(torch::kFloat16);
      self = self.to(torch::kFloat16);
    }

    compareKthvalueResults(
        resultValues, resultIndices, expectedValues, self, axis, keep_dims);
  }

  void testKthvalueValues(
      torch::IntArrayRef inputSize,
      torch::IntArrayRef outputSize,
      torch::ScalarType dtype,
      int k_value,
      int axis,
      bool keep_dims) {
    GenerateInputs(1, {inputSize}, dtype);

    auto resultValues = torch::empty(
        outputSize, torch::TensorOptions(dtype).device(torch::kHPU));
    auto resultIndices = torch::empty(
        outputSize, torch::TensorOptions(torch::kLong).device(torch::kHPU));
    torch::kthvalue_out(
        resultValues, resultIndices, GetHpuInput(0), k_value, axis, keep_dims);

    auto self = GetCpuInput(0);
    if (dtype == torch::kFloat16) {
      self = self.to(torch::kFloat);
    }

    auto expectedValues =
        torch::empty(outputSize, torch::TensorOptions(self.dtype()));
    auto expectedIndices =
        torch::empty(outputSize, torch::TensorOptions(torch::kLong));
    torch::kthvalue_out(
        expectedValues, expectedIndices, self, k_value, axis, keep_dims);
    if (dtype == torch::kFloat16) {
      expectedValues = expectedValues.to(torch::kFloat16);
      self = self.to(torch::kFloat16);
    }

    compareKthvalueResults(
        resultValues, resultIndices, expectedValues, self, axis, keep_dims);
  }
};

#define KTHVALUE_TESTS(TEST_NAME, DTYPE)                  \
  TEST_F(HpuOpTest, TEST_NAME) {                          \
    if (!IsDtypeSupportedOnCurrentDevice(DTYPE)) {        \
      GTEST_SKIP();                                       \
    }                                                     \
    if (isGaudi3()) {                                     \
      GTEST_SKIP() << "Test skipped on Gaudi3.";          \
    }                                                     \
    testKthvalue({3, 4}, DTYPE, 2, 0, true);              \
    testKthvalue({3, 4}, DTYPE, 2, 0, false);             \
    testKthvalue({3, 4, 6, 8}, DTYPE, 3, 3, true);        \
    testKthvalue({3, 4, 6, 8}, DTYPE, 3, 3, false);       \
    testKthvalue({3, 4, 5, 2, 3, 2}, DTYPE, 4, 2, false); \
    testKthvalue({3, 4, 5, 2, 3, 2}, DTYPE, 4, 2, true);  \
  }

#define KTHVALUE_VALUES_TESTS(TEST_NAME, DTYPE)                         \
  TEST_F(HpuOpTest, TEST_NAME) {                                        \
    if (!IsDtypeSupportedOnCurrentDevice(DTYPE)) {                      \
      GTEST_SKIP();                                                     \
    }                                                                   \
    if (isGaudi3()) {                                                   \
      GTEST_SKIP() << "Test skipped on Gaudi3.";                        \
    }                                                                   \
    testKthvalueValues({3, 4}, {1, 4}, DTYPE, 2, 0, true);              \
    testKthvalueValues({3, 4}, {1, 4}, DTYPE, 2, 0, false);             \
    testKthvalueValues({3, 4, 6, 8}, {3, 4, 6, 1}, DTYPE, 3, 3, true);  \
    testKthvalueValues({3, 4, 6, 8}, {3, 4, 6, 1}, DTYPE, 3, 3, false); \
    testKthvalueValues(                                                 \
        {3, 4, 5, 2, 3, 2}, {3, 4, 1, 2, 3, 2}, DTYPE, 4, 2, false);    \
    testKthvalueValues(                                                 \
        {3, 4, 5, 2, 3, 2}, {3, 4, 1, 2, 3, 2}, DTYPE, 4, 2, true);     \
  }

KTHVALUE_TESTS(kthvalue_float, torch::kFloat)
KTHVALUE_TESTS(kthvalue_bfloat16, torch::kBFloat16)
KTHVALUE_TESTS(kthvalue_float16, torch::kFloat16)
KTHVALUE_TESTS(kthvalue_int32, torch::kInt32)

KTHVALUE_VALUES_TESTS(kthvalue_values_float, torch::kFloat)
KTHVALUE_VALUES_TESTS(kthvalue_values_bfloat16, torch::kBFloat16)
KTHVALUE_VALUES_TESTS(kthvalue_values_float16, torch::kFloat16)
KTHVALUE_VALUES_TESTS(kthvalue_values_int32, torch::kInt32)