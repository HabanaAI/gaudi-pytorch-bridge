/**
 * Copyright (c) 2025 Intel Corporation
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
#include <torch/torch.h>
#include <algorithm>
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;

using ScalesMapRecord = std::pair<ScalarValueTypePair, ScalesIdxPair>;

class LazyH2dScalesTest : public habana_lazy_test::LazyTest {
 protected:
  /**
   * Creates H2D tensors with all possible hw-aligned scales and their
   * inversions, so they are picked in runtime by fp8 ops supporting H2D scales
   * instead of being created each time. Default exp_bias is 7 (only
   * torch.float8_e4m3fn is supported). Possible exp_bias for gaudi2 is [3, 7,
   * 11, 15], so -1 is added as an inversion of 15. Possible exp_bias for gaudi3
   * is [0, 63], so with inversions it's [-49, 63]
   */
  std::vector<double> generate_expected_scales() {
    static constexpr int default_bias =
        7; // Only torch.float8_e4m3fn supports H2D scales.
    std::vector<int> biases;
    if (habana::HPUDeviceContext::get_device().type() == synDeviceGaudi2) {
      biases = {-1, 3, 7, 11, 15};
    } else {
      biases.resize(113);
      std::iota(biases.begin(), biases.end(), -49);
    }

    std::vector<double> expected_scales;
    expected_scales.reserve(biases.size());
    for (const auto bias : biases) {
      expected_scales.push_back(std::pow(2.0, default_bias - bias));
    }

    return expected_scales;
  }

  double get_scale_from_pointer(const at::Tensor& scale_tensor) {
    auto hl_scale_tensor =
        habana_lazy::GetOrCreateHbLazyTensor(scale_tensor, at::kHPU)
            .CurrentTensorAttached()
            .value();
    auto tmeta{habana::get_tensor_extra_meta(hl_scale_tensor)};
    void* scale_ptr = tmeta->get_host_ptr();
    double scale_val = 0.0;
    if (scale_tensor.scalar_type() == at::ScalarType::Float) {
      scale_val = *reinterpret_cast<float*>(scale_ptr);
    } else {
      scale_val =
          static_cast<float>(*reinterpret_cast<at::BFloat16*>(scale_ptr));
    }
    return scale_val;
  }

  double validate_and_get_scale_from_map(
      const ScalesMapRecord& scales_record,
      std::optional<at::Tensor> scale_tensor = std::nullopt) {
    const auto key_scale = scales_record.first.value;
    const auto key_dtype = scales_record.first.dtype;
    const auto was_scale_used = scale_tensor.has_value() and
        (scale_tensor.value().item().toDouble() == key_scale) and
        (scale_tensor.value().scalar_type() == key_dtype);
    const auto& allocated_scale_tensor_vec = scales_record.second.scales;
    const auto scale_idx = scales_record.second.current_idx;

    EXPECT_EQ(allocated_scale_tensor_vec.size(), 1)
        << "Preallocated scales vector for each scale value should initially contain 1 element.";
    if (was_scale_used) {
      EXPECT_EQ(scale_idx, std::numeric_limits<decltype(scale_idx)>::max())
          << "Preallocated scale idx for used scale should be equal to "
          << std::numeric_limits<decltype(scale_idx)>::max() << ".";
    } else {
      EXPECT_EQ(scale_idx, 0)
          << "Preallocated scale idx should be initially equal to 0.";
    }

    const auto host_ptr_scale =
        get_scale_from_pointer(allocated_scale_tensor_vec[0]);
    EXPECT_EQ(host_ptr_scale, key_scale)
        << "Host_ptr of allocated scale is not equal to the actual scale value.";

    return key_scale;
  }

  void call_cast_to_fp8_op(
      const at::Tensor& input,
      const at::Tensor& cpu_scale) {
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("hpu::cast_to_fp8_v2", "")
                         .typed<std::tuple<at::Tensor, at::Tensor>(
                             const at::Tensor&,
                             const std::optional<at::Tensor>&,
                             bool,
                             bool,
                             at::ScalarType,
                             at::OptionalIntArrayRef)>();

    op.call(
        input,
        cpu_scale,
        false,
        false,
        at::ScalarType::Float8_e4m3fn,
        std::nullopt);
  }

  void TearDown() override {
    habana::HPUDeviceContext::set_scale_attributes(false, 0);
  }

  void validate_scales_map(
      std::optional<at::Tensor> scale_tensor = std::nullopt) {
    const auto expected_scales = generate_expected_scales();
    const auto scales_count = expected_scales.size();
    const auto& scales_map = habana_lazy::get_device_lazy_execution_context()
                                 ->getScalarToH2dScalesMapRef();

    EXPECT_EQ(scales_map.size(), 2 * scales_count)
        << "Expected the H2D scales map size to be twice the size of expected_scales (for float and bfloat16).";

    std::vector<double> float_scales;
    float_scales.reserve(scales_count);
    std::vector<double> bfloat16_scales;
    bfloat16_scales.reserve(scales_count);

    for (const auto& pair : scales_map) {
      const auto allocated_scale =
          validate_and_get_scale_from_map(pair, scale_tensor);
      if (pair.first.dtype == at::ScalarType::Float) {
        float_scales.push_back(allocated_scale);
      } else {
        bfloat16_scales.push_back(allocated_scale);
      }
    }

    std::sort(float_scales.begin(), float_scales.end(), std::greater<double>());
    std::sort(
        bfloat16_scales.begin(), bfloat16_scales.end(), std::greater<double>());

    EXPECT_EQ(float_scales, expected_scales)
        << "Preallocated float H2D scales are not equal to expected scales.";
    EXPECT_EQ(bfloat16_scales, expected_scales)
        << "Preallocated float H2D scales are not equal to expected scales.";
  }
};

TEST_F(LazyH2dScalesTest, InitialH2dScalesMap) {
  // Execute a simple operation to trigger device initialization.
  torch::sin(torch::rand({4}).to(torch::kHPU));

  const auto& initial_map = habana_lazy::get_device_lazy_execution_context()
                                ->getScalarToH2dScalesMapRef();
  EXPECT_TRUE(initial_map.empty())
      << "Expected the initial H2D scales map to be empty.";
}

TEST_F(LazyH2dScalesTest, H2dScalesMapWithoutGraphAttributes) {
  auto input = torch::randn({4, 8}).to(torch::kHPU);
  auto cpu_scale = torch::tensor(1.5, at::TensorOptions().dtype(at::kFloat));

  call_cast_to_fp8_op(input, cpu_scale);

  const auto& initial_map = habana_lazy::get_device_lazy_execution_context()
                                ->getScalarToH2dScalesMapRef();
  EXPECT_TRUE(initial_map.empty())
      << "Expected the initial H2D scales map to be empty when op called without setting scale attributes.";
}

TEST_F(LazyH2dScalesTest, H2dScalesMapWithNonHwScale) {
  // Execute a simple operation to trigger device initialization.
  torch::sin(torch::rand({4}).to(torch::kHPU));

  // Set the scale attributes to enable H2D scales feature.
  habana::HPUDeviceContext::set_scale_attributes(true, 43);

  auto input = torch::randn({4, 8}).to(torch::kHPU);
  auto non_hw_cpu_scale =
      torch::tensor(1.5, at::TensorOptions().dtype(at::kFloat));

  call_cast_to_fp8_op(input, non_hw_cpu_scale);
  validate_scales_map();
}

TEST_F(LazyH2dScalesTest, H2dScalesMapWithHwScale) {
  // Execute a simple operation to trigger device initialization.
  torch::sin(torch::rand({4}).to(torch::kHPU));

  // Set the scale attributes to enable H2D scales feature.
  habana::HPUDeviceContext::set_scale_attributes(true, 43);

  auto input = torch::randn({4, 8}).to(torch::kHPU);
  auto hw_cpu_scale =
      torch::tensor(16.0, at::TensorOptions().dtype(at::kFloat));

  // Call the op with cpu hw aligned scale. This should fill H2D scales map and
  // set the current index of the scale value to -1.
  call_cast_to_fp8_op(input, hw_cpu_scale);
  validate_scales_map(hw_cpu_scale);

  // Call the op with cpu hw aligned scale second time. This should create and
  // allocate a new H2D scale and push it to the scales vector for the scale
  // value. Current index value should still be -1.
  call_cast_to_fp8_op(input, hw_cpu_scale);

  const auto& scales_map = habana_lazy::get_device_lazy_execution_context()
                               ->getScalarToH2dScalesMapRef();
  const auto& used_scale_record = scales_map.at(
      {hw_cpu_scale.item().toDouble(), hw_cpu_scale.scalar_type()});
  const auto& used_scale_tensor_vec = used_scale_record.scales;
  const auto used_scale_idx = used_scale_record.current_idx;
  EXPECT_EQ(used_scale_tensor_vec.size(), 2)
      << "Preallocated scales vector for scale value used twice should contain 2 elements.";
  EXPECT_EQ(used_scale_idx, -1) << "Current index of used scale should be -1.";
  EXPECT_EQ(
      get_scale_from_pointer(used_scale_tensor_vec[0]),
      get_scale_from_pointer(used_scale_tensor_vec[1]))
      << "Preallocated scales vector for scale value used twice should contain 2 identical elements.";

  // Force mark step. This should set the current index of the used scale to 1
  // (last index of scales vector).
  HbLazyTensor::StepMarker({});

  const auto& used_scale_record_after_mark = scales_map.at(
      {hw_cpu_scale.item().toDouble(), hw_cpu_scale.scalar_type()});
  const auto& used_scale_tensor_vec_after_mark =
      used_scale_record_after_mark.scales;
  const auto used_scale_idx_after_mark =
      used_scale_record_after_mark.current_idx;
  EXPECT_EQ(used_scale_tensor_vec_after_mark.size(), 2)
      << "Preallocated scales vector for scale value used twice should contain 2 elements.";
  EXPECT_EQ(used_scale_idx_after_mark, 1)
      << "Current index of used scale should be 1 after mark step.";
  EXPECT_EQ(
      get_scale_from_pointer(used_scale_tensor_vec_after_mark[0]),
      get_scale_from_pointer(used_scale_tensor_vec_after_mark[1]))
      << "Preallocated scales vector for scale value used twice should contain 2 identical elements after mark step.";
}
