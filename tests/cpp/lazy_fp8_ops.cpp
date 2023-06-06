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

#include <cmath>
#include "backend/habana_device/HPUGuardImpl.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "hpu_ops/util.h"

bool IsUnsupported() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  const auto device_type = habana::HPURegistrar::get_device().type();
  return device_type == synDeviceGaudi;
}

at::Tensor simulateFp8Precision(const at::Tensor& input) {
  auto dtype = input.scalar_type();
  auto int_type = torch::kInt;
  auto mask = torch::tensor({2145386496}, int_type);
  auto mask_round = torch::tensor({1048575}, int_type);
  auto excessive_bits = torch::tensor({21}, int_type);
  if (dtype != torch::kFloat) {
    int_type = torch::kShort;
    mask = torch::tensor({32736}, int_type);
    mask_round = torch::tensor({15}, int_type);
    excessive_bits = torch::tensor({5}, int_type);
  }

  auto signs = torch::where(input < 0.0, -1.0, 1.0).to(dtype);
  auto as_int = input.view(int_type);
  auto mant_odd = torch::bitwise_and(
      torch::bitwise_right_shift(as_int, excessive_bits),
      torch::tensor({1}, int_type));
  auto as_int_masked = as_int + mask_round;
  auto as_int_odded = as_int_masked + mant_odd;
  auto masked = torch::bitwise_and(as_int_odded, mask);
  return masked.view(dtype) * signs;
}

class Fp8GeluTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<
          std::tuple<c10::optional<float>, c10::ScalarType, bool, bool>> {
 public:
  void Fp8GeluV2(
      torch::IntArrayRef input_shape,
      c10::optional<float> scale_opt,
      c10::ScalarType dtype,
      bool stochastic,
      bool is_amax) {
    auto input = (torch::rand(input_shape) * 30.0 + 10.0).to(dtype);

    float scale_val = scale_opt ? *scale_opt : 1.0;
    auto scale = torch::tensor({scale_val});
    auto scale_inv = scale.reciprocal();

    c10::optional<at::Tensor> scale_hpu = scale_opt
        ? c10::make_optional<at::Tensor>(scale.to("hpu"))
        : c10::nullopt;

    c10::optional<at::Tensor> scale_inv_hpu = scale_opt
        ? c10::make_optional<at::Tensor>(scale_inv.to("hpu"))
        : c10::nullopt;

    auto gelu = torch::gelu(input, "tanh");
    auto scaled_gelu_low_precision =
        simulateFp8Precision(gelu * scale.to(dtype));
    auto result_cpu = scaled_gelu_low_precision * scale_inv.to(dtype);
    auto retain_cpu = torch::tanh(
                          torch::sqrt(torch::tensor({2 / M_PI})) *
                          (input + 0.044715 * torch::pow(input, 3)))
                          .to(dtype);

    const auto [gelu_scaled, retain, amax] =
        fp8_gelu_v2_wrap(input.to("hpu"), scale_hpu, stochastic, is_amax);
    auto gelu_unscaled = cast_from_fp8_wrap(gelu_scaled, scale_inv_hpu, dtype);

    double rtol = stochastic ? 0.26 : 0.0;
    double atol = 0.01;

    Compare(result_cpu, gelu_unscaled, rtol, atol);
    if (is_amax) {
      Compare(torch::max(input.abs()).reshape({1}), amax.to(dtype), 0.0, 0.0);
    }
    Compare(retain_cpu, retain, 0.0, 0.0);
  }
};

TEST_P(Fp8GeluTest, fp8_gelu_v2) {
  if (IsUnsupported()) {
    GTEST_SKIP();
  }
  const auto& testParams = GetParam();
  const auto scale = std::get<0>(testParams);
  const auto dtype = std::get<1>(testParams);
  const auto stochastic = std::get<2>(testParams);
  const auto is_amax = std::get<3>(testParams);
  Fp8GeluV2({6, 12}, scale, dtype, stochastic, is_amax);
}

INSTANTIATE_TEST_SUITE_P(
    fp8_gelu_v2,
    Fp8GeluTest,
    ::testing::Combine(
        ::testing::Values<c10::optional<float>>(c10::nullopt, 0.75),
        ::testing::Values<c10::ScalarType>(torch::kFloat, torch::kBFloat16),
        ::testing::Values<bool>(true, false),
        ::testing::Values<bool>(true, false)));
