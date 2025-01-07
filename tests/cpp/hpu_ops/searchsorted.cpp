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

class HpuSearchSortedOpTest
    : public HpuOpTestUtil,
      public ::testing::WithParamInterface<testing::tuple<
          std::tuple<std::vector<long int>, std::vector<long int>>,
          at::ScalarType,
          bool,
          bool>> {
 public:
  struct GetName {
    template <class ParamType>
    std::string operator()(
        const ::testing::TestParamInfo<ParamType>& info) const {
      std::stringstream ss;
      ss << "sorted_sequence_" << std::get<0>(std::get<0>(info.param))
         << "_input_" << std::get<1>(std::get<0>(info.param)) << "_dtype_"
         << std::get<1>(info.param) << "_out_int32_"
         << (std::get<2>(info.param) ? "true" : "false") << "_right_"
         << (std::get<3>(info.param) ? "true" : "false");
      auto name = ss.str();
      std::replace(name.begin(), name.end(), ' ', 'x');
      return name;
    }
  };

 private:
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  create_sorted_sequence(
      torch::IntArrayRef inputSize,
      torch::ScalarType dtype,
      bool create_sorter) {
    GenerateInputs(1, {inputSize}, dtype);
    auto [cpu_sorted, cpu_indices] = torch::sort(GetCpuInput(0));
    auto hpu_sorted = cpu_sorted.to(torch::kHPU);
    auto hpu_indices = cpu_indices.to(torch::kHPU);
    if (create_sorter) {
      return {GetHpuInput(0), hpu_indices, GetCpuInput(0), cpu_indices};
    } else {
      return {hpu_sorted, hpu_indices, cpu_sorted, cpu_indices};
    }
  }

  void parse_params() {
    inputSize = std::get<0>(::testing::get<0>(GetParam()));
    valueSize = std::get<1>(::testing::get<0>(GetParam()));
    dtype = ::testing::get<1>(GetParam());
    out_int32 = ::testing::get<2>(GetParam());
    right = ::testing::get<3>(GetParam());
  }

 public:
  void testSearchSorted() {
    parse_params();

    auto [hpu_sorted, hpu_indices, cpu_sorted, cpu_indices] =
        create_sorted_sequence(inputSize, dtype, false);
    at::Tensor cpu_value = torch::randn(valueSize).to(dtype);
    at::Tensor hpu_value = cpu_value.to(torch::kHPU);

    at::Tensor hpu_result =
        torch::searchsorted(hpu_sorted, hpu_value, out_int32, right);
    at::Tensor cpu_result =
        torch::searchsorted(cpu_sorted, cpu_value, out_int32, right);

    Compare(cpu_result, hpu_result);
  }

  void testSearchSortedScalar() {
    parse_params();

    auto [hpu_sorted, hpu_indices, cpu_sorted, cpu_indices] =
        create_sorted_sequence(inputSize, dtype, false);
    if (hpu_sorted.dim() != 1) {
      GTEST_SKIP()
          << "Scalar parameter is supported only if sorted_sequence.dim() is equal to 1";
    }
    at::Scalar scalar = 0;

    at::Tensor hpu_result =
        torch::searchsorted(hpu_sorted, scalar, out_int32, right);
    at::Tensor cpu_result =
        torch::searchsorted(cpu_sorted, scalar, out_int32, right);

    Compare(cpu_result, hpu_result);
  }

  void testSearchSortedSorter() {
    parse_params();

    auto [hpu_sorted, hpu_indices, cpu_sorted, cpu_indices] =
        create_sorted_sequence(inputSize, dtype, true);

    at::Tensor cpu_value = torch::randn(valueSize).to(dtype);
    at::Tensor hpu_value = cpu_value.to(torch::kHPU);

    at::Tensor hpu_result = torch::searchsorted(
        hpu_sorted, hpu_value, out_int32, right, c10::nullopt, hpu_indices);
    at::Tensor cpu_result = torch::searchsorted(
        cpu_sorted, cpu_value, out_int32, right, c10::nullopt, cpu_indices);

    Compare(cpu_result, hpu_result);
  }

  void testSearchSortedOut() {
    parse_params();

    auto [hpu_sorted, hpu_indices, cpu_sorted, cpu_indices] =
        create_sorted_sequence(inputSize, dtype, false);
    at::Tensor cpu_value = torch::randn(valueSize).to(dtype);
    at::Tensor hpu_value = cpu_value.to(torch::kHPU);

    auto dtype = out_int32 ? torch::kInt : torch::kLong;
    at::Tensor hpu_result = torch::zeros(valueSize, dtype).to(torch::kHPU);
    at::Tensor cpu_result = torch::zeros(valueSize, dtype);
    torch::searchsorted_out(
        hpu_result, hpu_sorted, hpu_value, out_int32, right);
    torch::searchsorted_out(
        cpu_result, cpu_sorted, cpu_value, out_int32, right);

    Compare(cpu_result, hpu_result);
  }

 private:
  at::ScalarType dtype;
  at::IntArrayRef inputSize;
  at::IntArrayRef valueSize;
  bool out_int32;
  bool right;
};

TEST_P(HpuSearchSortedOpTest, TestBasic) {
  testSearchSorted();
}

TEST_P(HpuSearchSortedOpTest, TestScalar) {
  testSearchSortedScalar();
}

TEST_P(HpuSearchSortedOpTest, TestOut) {
  testSearchSortedOut();
}

TEST_P(HpuSearchSortedOpTest, TestSorter) {
  testSearchSortedSorter();
}

INSTANTIATE_TEST_CASE_P(
    SearchSorted,
    HpuSearchSortedOpTest,
    ::testing::Combine(
        ::testing::Values(
            std::make_tuple(
                std::vector<long int>{3, 10},
                std::vector<long int>{3, 3}),
            std::make_tuple(
                std::vector<long int>{20},
                std::vector<long int>{5})),
        ::testing::Values(at::kFloat, at::kBFloat16, at::kLong, at::kInt),
        ::testing::Values(true, false),
        ::testing::Values(true, false)),
    HpuSearchSortedOpTest::GetName());
