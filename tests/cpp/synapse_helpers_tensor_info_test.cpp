/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include <absl/types/variant.h>
#include <gtest/gtest.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_info.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "backend/synapse_helpers/synapse_error.h"

TEST(SynapseHelpersTensorInfoTest, TensorInfoSize) {
  using namespace synapse_helpers;
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  int64_t shape = std::numeric_limits<uint32_t>::max();
  synGraphHandle h;
  auto& synapse_device = habana::HPURegistrar::get_device().syn_device();
  ASSERT_EQ(synSuccess, synGraphCreate(&h, synapse_device.type()));
  auto input_shape = tensor::shape_t{1_D, {shape}};
  auto build_result = tensor_builder(synDataType::syn_type_bf16)
                          .with_shape(input_shape)
                          .build(synapse_device, h);
  ASSERT_EQ(absl::holds_alternative<synapse_error>(build_result), false);
  auto tensor = get_value(std::move(build_result));
  PtTensorInfo tensor_info = PtTensorInfo(tensor, "ab");
  ASSERT_EQ(tensor_info.get_size(), 2 * shape);
}

TEST(SynapseHelpersTensorInfoTest, TensorInfoNumel) {
  using namespace synapse_helpers;
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  int64_t shape = std::numeric_limits<uint32_t>::max() * 2;
  synGraphHandle h;
  auto& synapse_device = habana::HPURegistrar::get_device().syn_device();
  ASSERT_EQ(synSuccess, synGraphCreate(&h, synapse_device.type()));
  auto input_shape = tensor::shape_t{1_D, {shape}};
  auto build_result = tensor_builder(synDataType::syn_type_bf16)
                          .with_shape(input_shape)
                          .build(synapse_device, h);
  ASSERT_EQ(absl::holds_alternative<synapse_error>(build_result), false);
  auto tensor = get_value(std::move(build_result));
  PtTensorInfo tensor_info = PtTensorInfo(tensor, "ab");
  ASSERT_EQ(tensor_info.get_numel(), shape);
}