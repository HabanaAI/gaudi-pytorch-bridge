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
#include <absl/types/variant.h>
#include <gtest/gtest.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>
#include "backend/habana_device/HPUDevice.h"
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
  auto& synapse_device = habana::HPUDeviceContext::get_device();
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
  auto& synapse_device = habana::HPUDeviceContext::get_device();
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