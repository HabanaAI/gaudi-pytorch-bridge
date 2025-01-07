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
#include <torch/torch.h>
#include <random>
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

class HpuOpTestUtilBase : public habana_lazy_test::EnvHelper {
 public:
  void Compare(
      const torch::Tensor& cpu_result,
      const torch::Tensor& hpu_result,
      c10::optional<double> rtol = c10::nullopt,
      c10::optional<double> atol = c10::nullopt) const;

  template <typename... Ts>
  void Compare(
      const std::tuple<Ts...>& cpu_result,
      const std::tuple<Ts...>& hpu_result,
      c10::optional<double> rtol = c10::nullopt,
      c10::optional<double> atol = c10::nullopt) const;

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

  template <typename... Ts, std::size_t... Is>
  void compareTuple(
      const std::tuple<Ts...>& expected,
      const std::tuple<Ts...>& result,
      std::index_sequence<Is...>,
      c10::optional<double> rtol,
      c10::optional<double> atol) const;
};

template <typename... Ts>
void HpuOpTestUtilBase::Compare(
    const std::tuple<Ts...>& expected,
    const std::tuple<Ts...>& result,
    c10::optional<double> rtol,
    c10::optional<double> atol) const {
  compareTuple(expected, result, std::index_sequence_for<Ts...>{}, rtol, atol);
}

template <typename... Ts, std::size_t... Is>
void HpuOpTestUtilBase::compareTuple(
    const std::tuple<Ts...>& expected,
    const std::tuple<Ts...>& result,
    std::index_sequence<Is...>,
    c10::optional<double> rtol,
    c10::optional<double> atol) const {
  (Compare(std::get<Is>(expected), std::get<Is>(result), rtol, atol), ...);
}

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
 public:
  template <typename T>
  static std::string SerializeShape(const std::vector<T> &, const std::string &);
 protected:
  static std::string FixTestName(std::string name);
 private:
  void SetUp() override {
    DisableCpuFallback();
    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
    TearDownBridge();
  }
  void TearDown() override {
    RestoreMode();
  }
};

template <typename T>
std::string HpuOpTestUtil::SerializeShape(const std::vector<T> &shape,
                                   const std::string &prefix) {
  std::string s = prefix;
  auto seprator = "";
  for (auto &&d : shape) {
    s += seprator;
    s += std::to_string(d);
    seprator = "x";
  }
  return s;
}

template <typename T>
class DTypeSupportTest : public testing::Test,
                         public testing::WithParamInterface<T> {
  void SetUp() override {
    clearRegisteredFallbacks();
  }
  void TearDown() override {
    clearRegisteredFallbacks();
  }

 private:
  void clearRegisteredFallbacks() {
    auto& op_fallback_frequency =
        habana::HpuFallbackHelper::get()->get_op_count();
    (const_cast<std::unordered_map<std::string, size_t>&>(
         op_fallback_frequency))
        .clear();
  }
};
