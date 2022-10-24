/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/torch.h>
#include <random>

class HpuOpTestUtilBase : public habana_lazy_test::EnvHelper {
 public:
  void Compare(
      const torch::Tensor& cpu_result,
      const torch::Tensor& hpu_result,
      double rtol = 1e-03,
      double atol = 1e-03) const;

  torch::Tensor& GetCpuInput(int index) {
    return m_cpu_inputs.at(index);
  }

  torch::Tensor& GetHpuInput(int index) {
    return m_hpu_inputs.at(index);
  }

  void GenerateInputs(int num_inputs) {
    GenerateInputs(num_inputs, {torch::kFloat}, {});
  }

  // Generate inputs with different dtypes per input
  void GenerateInputs(
      int num_inputs,
      torch::ArrayRef<torch::ScalarType> dtypes) {
    GenerateInputs(num_inputs, {}, dtypes);
  }

  // Generate inputs with different sizes per input
  void GenerateInputs(
      int num_inputs,
      torch::ArrayRef<torch::IntArrayRef> sizes) {
    GenerateInputs(num_inputs, sizes, {});
  }

  // sizes and dtypes can be in any order
  void GenerateInputs(
      int num_inputs,
      torch::ArrayRef<torch::ScalarType> dtypes,
      torch::ArrayRef<torch::IntArrayRef> sizes) {
    GenerateInputs(num_inputs, sizes, dtypes);
  }

  // Generate inputs with different dtypes/sizes per input
  void GenerateInputs(
      int num_inputs,
      torch::ArrayRef<torch::IntArrayRef> sizes_,
      torch::ArrayRef<torch::ScalarType> dtypes_);

  void GenerateIntInputs(
      int num_inputs,
      torch::ArrayRef<torch::IntArrayRef> sizes,
      int low,
      int high);

  template <typename T = float>
  T GenerateScalar(
      c10::optional<T> min = c10::nullopt,
      c10::optional<T> max = c10::nullopt) const;

 private:
  const std::vector<int64_t> m_dims = {4, 5, 6};
  std::vector<torch::Tensor> m_cpu_inputs;
  std::vector<torch::Tensor> m_hpu_inputs;
  mutable std::mt19937 m_mt;
};

template <typename T>
T HpuOpTestUtilBase::GenerateScalar(c10::optional<T> min, c10::optional<T> max)
    const {
  std::uniform_real_distribution<T> dist(min.value_or(-127), max.value_or(128));
  return dist(m_mt);
}

template <>
int HpuOpTestUtilBase::GenerateScalar(
    c10::optional<int> min,
    c10::optional<int> max) const;

template <>
bool HpuOpTestUtilBase::GenerateScalar(
    c10::optional<bool> min,
    c10::optional<bool> max) const;

class HpuOpTestUtil : public HpuOpTestUtilBase, public ::testing::Test {
  void SetUp() override {
    DisableCpuFallback();
    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }
  void TearDown() override {
    RestoreMode();
  }
};
