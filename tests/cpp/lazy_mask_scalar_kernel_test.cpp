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

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyMaskKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyMaskKernelTest, MaskedScaleInplaceTest) {
  const std::vector<int64_t> dimensions{7, 3, 5};
  const int randomLimit = 300;
  torch::Tensor A = torch::randn(dimensions);
  torch::Tensor B = torch::randn(dimensions);

  // Generate random number for scalar
  float x = (float)rand() / (float)(RAND_MAX / randomLimit);
  double scale = rand() % 2 ? x : -1 * x;

  // Eager section:
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hExpected = at::_masked_scale(hA, hB, scale);
  Tensor expected = hExpected.to(torch::kCPU);

  // Lazy Section
  auto hAL = A.to(torch::kHPU);
  auto hBL = B.to(torch::kHPU);
  auto hOut = at::_masked_scale(hAL, hBL, scale);
  Tensor out = hOut.to(kCPU);

  EXPECT_EQ(allclose(out, expected), true);
}

TEST_F(LazyMaskKernelTest, MaskedFillInplaceTest) {
  const std::vector<int64_t> dimensions{3, 3};
  torch::Tensor A = torch::randn(dimensions);
  bool data[] = {true, false, false, false, true, false, false, false, true};
  torch::Tensor mask = torch::from_blob(data, dimensions).to(torch::kBool);

  torch::Tensor value = torch::randn({}); // Only 0-dim tensor accesped
  auto hA = A.to(torch::kHPU);
  auto cpuOut = A.masked_fill_(mask, value);

  auto hValue = value.to(torch::kHPU);
  auto hMask = mask.to(torch::kHPU);

  auto result = hA.masked_fill_(hMask, hValue);
  Tensor hOut = result.to(kCPU);
  EXPECT_TRUE(allclose(hOut, cpuOut));
}

TEST_F(LazyMaskKernelTest, MaskedFillScalarInplaceTest) {
  const std::vector<int64_t> dimensions{3, 3};
  torch::Tensor A = torch::randn(dimensions);
  bool data[] = {true, false, false, false, true, false, false, false, true};
  torch::Tensor mask = torch::from_blob(data, dimensions).to(torch::kBool);
  Scalar value = 35;

  auto hA = A.to(torch::kHPU);
  auto cpuOut = A.masked_fill_(mask, value);

  auto hMask = mask.to(torch::kHPU);
  auto result = hA.masked_fill_(hMask, value);
  Tensor hOut = result.to(kCPU);
  EXPECT_TRUE(allclose(hOut, cpuOut));
}

TEST_F(LazyMaskKernelTest, MaskedFillScalarInplaceBf16Test) {
  const std::vector<int64_t> dimensions{3, 3};
  at::TensorOptions options(ScalarType::BFloat16);
  torch::Tensor A = torch::randn(dimensions, options);
  torch::Tensor B = torch::randn({1}, options);
  bool data[] = {true, false, false, false, true, false, false, false, true};
  torch::Tensor mask = torch::from_blob(data, dimensions).to(torch::kBool);
  Scalar value = B.item();

  auto hA = A.to(torch::kHPU);
  auto cpuOut = A.masked_fill_(mask, value);

  auto hMask = mask.to(torch::kHPU);
  auto result = hA.masked_fill_(hMask, value);
  Tensor hOut = result.to(kCPU);
  EXPECT_TRUE(allclose(hOut, cpuOut));
}

TEST_F(LazyMaskKernelTest, MaskedFillScalarInplaceIntTest) {
  const std::vector<int64_t> dimensions{3, 3};
  at::TensorOptions options(ScalarType::Int);
  torch::Tensor A = torch::randint(100, dimensions, options);
  torch::Tensor B = torch::randint(100, {1}, options);
  bool data[] = {true, false, false, false, true, false, false, false, true};
  torch::Tensor mask = torch::from_blob(data, dimensions).to(torch::kBool);
  Scalar value = B.item();

  auto hA = A.to(torch::kHPU);
  auto cpuOut = A.masked_fill_(mask, value);

  auto hMask = mask.to(torch::kHPU);
  auto result = hA.masked_fill_(hMask, value);
  Tensor hOut = result.to(kCPU);
  EXPECT_TRUE(allclose(hOut, cpuOut));
}

TEST_F(LazyMaskKernelTest, MaskedFillScalarInplaceDoubleTest) {
  const std::vector<int64_t> dimensions{3, 3};
  at::TensorOptions options(ScalarType::Double);
  torch::Tensor A = torch::randn(dimensions, options);
  torch::Tensor B = torch::randn({1}, options);
  bool data[] = {true, false, false, false, true, false, false, false, true};
  torch::Tensor mask = torch::from_blob(data, dimensions).to(torch::kBool);
  Scalar value = B.item();

  auto hA = A.to(torch::kHPU);
  auto cpuOut = A.masked_fill_(mask, value);

  auto hMask = mask.to(torch::kHPU);
  auto result = hA.masked_fill_(hMask, value);
  Tensor hOut = result.to(kCPU);
  EXPECT_TRUE(allclose(hOut, cpuOut));
}

TEST_F(LazyMaskKernelTest, MaskedFillScalarInplaceLongTest) {
  const std::vector<int64_t> dimensions{3, 3};
  at::TensorOptions options(ScalarType::Long);
  torch::Tensor A = torch::randint(100, dimensions, options);
  torch::Tensor B = torch::randint(100, {1}, options);
  bool data[] = {true, false, false, false, true, false, false, false, true};
  torch::Tensor mask = torch::from_blob(data, dimensions).to(torch::kBool);
  Scalar value = B.item();

  auto hA = A.to(torch::kHPU);
  auto cpuOut = A.masked_fill_(mask, value);

  auto hMask = mask.to(torch::kHPU);
  auto result = hA.masked_fill_(hMask, value);
  Tensor hOut = result.to(kCPU);
  EXPECT_TRUE(allclose(hOut, cpuOut));
}

TEST_F(LazyMaskKernelTest, MaskedSelectTest) {
  const std::vector<int64_t> dimensions{3, 3};
  torch::Tensor A = torch::randn(dimensions);
  torch::Tensor mask = A.ge(0.5);
  auto cpuOut = torch::masked_select(A, mask);

  auto hA = A.to(torch::kHPU);
  auto hMask = mask.to(torch::kHPU);
  auto result = torch::masked_select(hA, hMask);
  Tensor hOut = result.to(kCPU);

  EXPECT_TRUE(allclose(hOut, cpuOut));
}

#define MASKED_FILL_SCALAR_DIFFERENT_DTYPES_TEST(A_dtype, value_dtype)        \
  TEST_F(                                                                     \
      LazyMaskKernelTest,                                                     \
      MaskedFillScalarInplace##A_dtype##_##value_dtype##Test) {               \
    const std::vector<int64_t> dimensions{3, 3};                              \
    torch::Tensor A = torch::randn(dimensions).to(ScalarType::A_dtype);       \
    torch::Tensor B = torch::randn({1}).to(ScalarType::value_dtype);          \
    bool data[] = {                                                           \
        true, false, false, false, true, false, false, false, true};          \
    torch::Tensor mask = torch::from_blob(data, dimensions).to(torch::kBool); \
    Scalar value = B.item();                                                  \
                                                                              \
    auto hA = A.to(torch::kHPU);                                              \
    auto cpuOut = A.masked_fill_(mask, value);                                \
                                                                              \
    auto hMask = mask.to(torch::kHPU);                                        \
    auto result = hA.masked_fill_(hMask, value);                              \
    Tensor hOut = result.to(torch::kCPU);                                     \
    EXPECT_TRUE(allclose(hOut, cpuOut));                                      \
  }

#define MASKED_FILL_VECTOR_DIFFERENT_DTYPES_TEST(A_dtype, value_dtype)        \
  TEST_F(                                                                     \
      LazyMaskKernelTest,                                                     \
      MaskedFillTensorInplace##A_dtype##_##value_dtype##Test) {               \
    const std::vector<int64_t> dimensions{3, 3};                              \
    torch::Tensor A = torch::randn(dimensions).to(ScalarType::A_dtype);       \
    torch::Tensor B = torch::tensor(5.01).to(ScalarType::value_dtype);        \
    bool data[] = {                                                           \
        true, false, false, false, true, false, false, false, true};          \
    torch::Tensor mask = torch::from_blob(data, dimensions).to(torch::kBool); \
                                                                              \
    auto hA = A.to(torch::kHPU);                                              \
    auto cpuOut = A.masked_fill_(mask, B);                                    \
                                                                              \
    auto hMask = mask.to(torch::kHPU);                                        \
    auto result = hA.masked_fill_(hMask, B);                                  \
    Tensor hOut = result.to(torch::kCPU);                                     \
    EXPECT_TRUE(allclose(hOut, cpuOut));                                      \
  }

#define MASKED_FILL_DIFFERENT_DTYPES_TEST(A_dtype, value_dtype)  \
  MASKED_FILL_VECTOR_DIFFERENT_DTYPES_TEST(A_dtype, value_dtype) \
  MASKED_FILL_SCALAR_DIFFERENT_DTYPES_TEST(A_dtype, value_dtype)

MASKED_FILL_DIFFERENT_DTYPES_TEST(BFloat16, Float)
MASKED_FILL_DIFFERENT_DTYPES_TEST(Float, BFloat16)
MASKED_FILL_DIFFERENT_DTYPES_TEST(Float, Int)
MASKED_FILL_DIFFERENT_DTYPES_TEST(BFloat16, Char)
MASKED_FILL_DIFFERENT_DTYPES_TEST(Char, Int)
MASKED_FILL_DIFFERENT_DTYPES_TEST(Int, Char)