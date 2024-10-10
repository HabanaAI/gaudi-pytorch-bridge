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

using GroupNormSizes_t =
    std::tuple<long int, long int, long int, long int, long int>;

class NativeGroupNormTests
    : public HpuOpTestUtil,
      public ::testing::WithParamInterface<
          std::tuple<GroupNormSizes_t, float, torch::ScalarType, bool>> {
  void SetUp() override {
    const auto flagParam = std::get<3>(GetParam());
    SET_ENV_FLAG_NEW(PT_HPU_USE_BN_FWD_IN_GN_BWD, flagParam, 1);
  }

  void TearDown() override {
    UNSET_ENV_FLAG_NEW(PT_HPU_USE_BN_FWD_IN_GN_BWD);
  }

 public:
  struct GetName {
    template <class ParamType>
    std::string operator()(
        const ::testing::TestParamInfo<ParamType>& info) const {
      ::std::stringstream ss;

      const auto [groupNormSizes, epsilon, dtype, flagParam] = info.param;
      const auto [N, C, H, W, G] = groupNormSizes;
      ss << N << 'x' << C << 'x' << H << 'x' << W << '_' << G << '_';
      ss << "eps_" << epsilon << "_" << dtype << "_"
         << "PT_HPU_USE_BN_FWD_IN_GN_BWD_" << flagParam;

      std::string testName = ss.str();
      std::replace(testName.begin(), testName.end(), '-', '_');
      std::replace(testName.begin(), testName.end(), '.', 'p');

      return testName;
    }
  };
};

TEST_P(NativeGroupNormTests, GroupNormFwdBwdExecute) {
  constexpr const auto OUTPUTS_NUMBER = 3;
  const auto [groupNormSizes, epsilon, dtype, flagParam] = GetParam();

  const auto [N, C, H, W, G] = groupNormSizes;

  const auto [atol, rtol] = dtype == torch::kFloat32
      ? std::make_tuple(c10::nullopt, c10::nullopt)
      : std::make_tuple(
            c10::optional<double>(0.05), c10::optional<double>(0.05));

  // Input, weight, bias, grad
  GenerateInputs(4, {{N, C, H, W}, {C}, {C}, {N, C, H, W}}, {dtype});

  auto results_fwd_cpu = torch::native_group_norm(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), N, C, H * W, G, epsilon);
  auto results_fwd_hpu = torch::native_group_norm(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), N, C, H * W, G, epsilon);

  Compare(results_fwd_cpu, results_fwd_hpu, rtol, atol);

  auto results_bwd_cpu = torch::native_group_norm_backward(
      GetCpuInput(3),
      GetCpuInput(0),
      std::get<1>(results_fwd_cpu),
      std::get<2>(results_fwd_cpu),
      GetCpuInput(1),
      N,
      C,
      H * W,
      G,
      {true, true, true});

  auto results_bwd_hpu = torch::native_group_norm_backward(
      GetHpuInput(3),
      GetHpuInput(0),
      std::get<1>(results_fwd_hpu),
      std::get<2>(results_fwd_hpu),
      GetHpuInput(1),
      N,
      C,
      H * W,
      G,
      {true, true, true});

  Compare(results_bwd_cpu, results_bwd_hpu, rtol, atol);
}

INSTANTIATE_TEST_CASE_P(
    NativeGroupNormSuite,
    NativeGroupNormTests,
    ::testing::Combine(
        ::testing::Values(
            std::make_tuple(16, 8, 2, 4, 8),
            std::make_tuple(4, 8, 3, 5, 2)), // N, C, H, W, G
        ::testing::Values(0.00005, 0.00001),
        ::testing::Values(at::kFloat, at::kBFloat16),
        ::testing::Bool()),
    NativeGroupNormTests::GetName());
