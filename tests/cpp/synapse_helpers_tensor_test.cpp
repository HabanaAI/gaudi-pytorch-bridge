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
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/tensor_builder.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "backend/synapse_helpers/synapse_error.h"

TEST(SynapseHelpersTest, NonDynamicTensorBuilding) {
  using namespace synapse_helpers;
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  auto& synapse_device = habana::HPUDeviceContext::get_device();
  synGraphHandle h;
  ASSERT_EQ(synSuccess, synGraphCreate(&h, synapse_device.type()));
  auto input_shape = tensor::shape_t{5_D, {5, 4, 3, 2, 1}};
  auto type = habana_helpers::pytorch_to_synapse_type(c10::ScalarType::Float);
  auto build_result =
      tensor_builder(input_shape, type).build(synapse_device, h);
  ASSERT_EQ(absl::holds_alternative<synapse_error>(build_result), false);
  auto tensor = get_value(std::move(build_result));
  EXPECT_EQ(tensor.num_elements(), 120);
  ASSERT_EQ(tensor.shape(), input_shape);
  ASSERT_NE(tensor.shape(), tensor::shape_t(5_D, {1, 2, 3, 4, 5}));
  ASSERT_NE(tensor.shape(), tensor::shape_t(4_D, {4, 3, 2, 1}));
  ASSERT_NE(tensor.shape(), tensor::shape_t(5_D, {5, 4, 3, 2, 0}));
  ASSERT_FALSE(tensor.has_dynamic_shape());
  ASSERT_EQ(synSuccess, synGraphDestroy(h));
}

TEST(SynapseHelpersTest, NonDynamicTensorWithShape) {
  using namespace synapse_helpers;
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  synGraphHandle h;
  auto& synapse_device = habana::HPUDeviceContext::get_device();
  ASSERT_EQ(synSuccess, synGraphCreate(&h, synapse_device.type()));
  auto input_shape = tensor::shape_t{5_D, {5, 4, 3, 2, 1}};
  auto build_result = tensor_builder(synDataType::syn_type_float)
                          .with_shape(input_shape)
                          .build(synapse_device, h);
  ASSERT_EQ(absl::holds_alternative<synapse_error>(build_result), false);
  auto tensor = get_value(std::move(build_result));
  ASSERT_EQ(tensor.num_elements(), 120);
  ASSERT_EQ(tensor.type(), synDataType::syn_type_float);
  ASSERT_EQ(tensor.shape(), input_shape);
  ASSERT_FALSE(tensor.has_dynamic_shape());
  ASSERT_EQ(synSuccess, synGraphDestroy(h));
}

TEST(SynapseHelpersTest, NonDynamicTensorWithRank) {
  using namespace synapse_helpers;
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  synGraphHandle h;
  auto& synapse_device = habana::HPUDeviceContext::get_device();
  ASSERT_EQ(synSuccess, synGraphCreate(&h, synapse_device.type()));
  auto input_shape = tensor::shape_t{3_D, {1, 2, 3}};
  auto build_result = tensor_builder(input_shape)
                          .with_rank_at_least(5)
                          .build(synapse_device, h);
  ASSERT_EQ(absl::holds_alternative<synapse_error>(build_result), false);
  auto tensor = get_value(std::move(build_result));
  ASSERT_EQ(tensor.num_elements(), 6);
  ASSERT_EQ(tensor.type(), synDataType::syn_type_float);
  ASSERT_NE(tensor.shape(), input_shape);
  ASSERT_EQ(tensor.shape(), tensor::shape_t(5_D, {1, 2, 3, 1, 1}));
  ASSERT_EQ(tensor.shape().rank(), 5_D);
  ASSERT_FALSE(tensor.has_dynamic_shape());
  ASSERT_EQ(synSuccess, synGraphDestroy(h));
}

TEST(SynapseHelpersTest, DynamicTensorBuilding) {
  using namespace synapse_helpers;
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  synGraphHandle h;
  auto& synapse_device = habana::HPUDeviceContext::get_device();
  ASSERT_EQ(synSuccess, synGraphCreate(&h, synapse_device.type()));
  auto min = tensor::shape_t{5_D, {5, 4, 3, 2, 1}};
  auto max = tensor::shape_t{5_D, {10, 4, 6, 2, 1}};
  auto dynamic_shape = tensor::dynamic_shape_t{min, max};
  auto build_result = tensor_builder(synDataType::syn_type_float)
                          .with_dynamic_shape(dynamic_shape)
                          .build(synapse_device, h);
  ASSERT_EQ(absl::holds_alternative<synapse_error>(build_result), false);
  auto tensor = get_value(std::move(build_result));
  ASSERT_EQ(tensor.num_elements(), 480);
  ASSERT_EQ(tensor.type(), synDataType::syn_type_float);
  ASSERT_EQ(tensor.dynamic_shape(), dynamic_shape);
  ASSERT_EQ(tensor.shape(), max);
  ASSERT_NE(
      tensor.dynamic_shape(),
      tensor::dynamic_shape_t(
          tensor::shape_t{5_D, {5, 4, 3, 2, 1}},
          tensor::shape_t{5_D, {9, 4, 6, 2, 1}}));
  ASSERT_EQ(tensor.dynamic_shape().min(), min);
  ASSERT_EQ(tensor.dynamic_shape().max(), max);
  ASSERT_EQ(tensor.dynamic_shape().rank(), 5_D);
  ASSERT_EQ(tensor.dynamic_shape().min().rank(), 5_D);
  ASSERT_EQ(tensor.dynamic_shape().max().rank(), 5_D);
  ASSERT_TRUE(tensor.has_dynamic_shape());
  ASSERT_EQ(synSuccess, synGraphDestroy(h));
}

TEST(SynapseHelpersTest, DynamicTensorWithRank) {
  using namespace synapse_helpers;
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  synGraphHandle h;
  auto& synapse_device = habana::HPUDeviceContext::get_device();
  ASSERT_EQ(synSuccess, synGraphCreate(&h, synapse_device.type()));
  auto min = tensor::shape_t{5_D, {5, 4, 3, 2, 1}};
  auto max = tensor::shape_t{5_D, {10, 4, 6, 2, 1}};
  auto dynamic_shape = tensor::dynamic_shape_t{min, max};
  auto build_result = tensor_builder(synDataType::syn_type_float)
                          .with_dynamic_shape(dynamic_shape)
                          .build(synapse_device, h);
  ASSERT_EQ(absl::holds_alternative<synapse_error>(build_result), false);
  auto tensor = get_value(std::move(build_result));
  ASSERT_EQ(tensor.num_elements(), 480);
  ASSERT_EQ(tensor.type(), synDataType::syn_type_float);
  ASSERT_EQ(tensor.dynamic_shape(), dynamic_shape);
  ASSERT_EQ(tensor.shape(), max);
  ASSERT_EQ(tensor.shape().rank(), 5_D);
  ASSERT_TRUE(tensor.has_dynamic_shape());
  ASSERT_EQ(synSuccess, synGraphDestroy(h));
}

TEST(SynapseHelpersTest, DynamicShape) {
  using namespace synapse_helpers;
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  synGraphHandle h;
  auto& synapse_device = habana::HPUDeviceContext::get_device();
  ASSERT_EQ(synSuccess, synGraphCreate(&h, synapse_device.type()));
  auto min = tensor::shape_t{2_D, {5, 4}};
  auto max = tensor::shape_t{2_D, {10, 4}};
  auto dynamic_shape = tensor::dynamic_shape_t{min, max};
  dynamic_shape.set_rank(5_D);
  ASSERT_EQ(dynamic_shape.min(), tensor::shape_t(5_D, {5, 4, 1, 1, 1}));
  ASSERT_EQ(dynamic_shape.max(), tensor::shape_t(5_D, {10, 4, 1, 1, 1}));
  dynamic_shape.set_dim(1, 5);
  ASSERT_EQ(dynamic_shape.min(), tensor::shape_t(5_D, {5, 5, 1, 1, 1}));
  ASSERT_EQ(dynamic_shape.max(), tensor::shape_t(5_D, {10, 5, 1, 1, 1}));
  dynamic_shape.set_dim(2, 4, 10);
  ASSERT_EQ(dynamic_shape.min(), tensor::shape_t(5_D, {5, 5, 4, 1, 1}));
  ASSERT_EQ(dynamic_shape.max(), tensor::shape_t(5_D, {10, 5, 10, 1, 1}));
  ASSERT_EQ(synSuccess, synGraphDestroy(h));
}
