/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "habana_kernels/nonzero_kernel.h"
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>
#include <limits>
#include "backend/create_pt_tensor.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/hpu_op_helper.h"

using namespace torch;
using namespace habana;

using namespace std::literals;

void NonZeroOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  HABANA_ASSERT(
      inputs.size() == 1,
      "Incorrect size of inputs expected for NonZero operator");
  HABANA_ASSERT(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for NonZero operator");

  auto self = inputs[0].toTensor();
  auto input_shape = self.sizes();
  const auto dimensions_size_t = input_shape.size();
  HABANA_ASSERT(
      dimensions_size_t <= static_cast<size_t>(std::numeric_limits<int>::max()),
      "dimensions is too large for int");
  const int dimensions = static_cast<int>(dimensions_size_t);

  const auto elements_int64 = self.numel();
  HABANA_ASSERT(
      elements_int64 >= 0 && elements_int64 <= std::numeric_limits<int>::max(),
      "elements is out of int range");
  const int elements = static_cast<int>(elements_int64);
  auto output_shape = DimVector{elements, dimensions};
  auto shape_tensor_shape = DimVector{5};
  // Create PT output stage 2
  auto cordinates_of_true = habana::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);
  auto shape_tensor = habana::createPTTensor(
      self,
      shape_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);

  std::vector<at::Tensor> outputs{cordinates_of_true, shape_tensor};
  HabanaOperator::SetPTOutputs(outputs);
}

float NonZeroOperator::round_dims(
    const at::Tensor& input_tensor,
    int group_size) {
  const auto group_size_f = static_cast<float>(group_size);
  const auto dim_int64 = input_tensor.dim();
  HABANA_ASSERT(dim_int64 >= 1, "tensor must have at least 1 dimension");

  const auto last_dim_idx = static_cast<size_t>(dim_int64 - 1);
  const auto last_dim_size_int64 = input_tensor.sizes()[last_dim_idx];
  const auto last_dim_size_f = static_cast<float>(last_dim_size_int64);

  const auto last_dim_rounded =
      std::ceil(last_dim_size_f / group_size_f) * group_size_f;
  return last_dim_rounded;
}

std::vector<int64_t> NonZeroOperator::compute_output_st_shape(
    const at::Tensor& input_tensor) {
  constexpr int group_size = 64;
  auto last_dim_rounded = NonZeroOperator::round_dims(input_tensor, group_size);
  auto out_st_shape = input_tensor.sizes().vec();
  auto group_size_aligned_dim =
      (long int)last_dim_rounded / (long int)group_size;
  out_st_shape.pop_back();
  out_st_shape.emplace_back(group_size_aligned_dim);
  out_st_shape.emplace_back(group_size);
  return out_st_shape;
}

std::vector<int64_t> NonZeroOperator::compute_output_shape(
    const at::Tensor& self) {
  auto input_shape = self.sizes();
  const auto dimensions_size_t = input_shape.size();
  HABANA_ASSERT(
      dimensions_size_t <= static_cast<size_t>(std::numeric_limits<int>::max()),
      "dimensions is too large for int");
  const int dimensions = static_cast<int>(dimensions_size_t);

  auto elements = self.numel();
  if ((self.dim() <= 4) and (self.dim() > 0)) {
    elements = 1;
    const auto last_dim_rounded = round_dims(self, 64);
    for (size_t i = 0; i < self.sizes().size() - 1; i++) {
      const auto size_int64 = self.sizes()[i];
      const auto size_f = static_cast<float>(size_int64);
      const auto elements_f = static_cast<float>(elements);
      elements = static_cast<int64_t>(elements_f * size_f);
    }
    const auto elements_f = static_cast<float>(elements);
    elements = static_cast<int64_t>(elements_f * last_dim_rounded);
  }
  std::vector<int64_t> output_shape{elements, dimensions};
  return output_shape;
}

InferOutputMetaRetType NonZeroOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  if (self.dim() > 4) {
    auto output_shape = compute_output_shape(self);
    std::vector<int64_t> shape_tensor_shape = {5};
    InferOutputMetaRetType out;
    auto metaData = TensorMetaData(
        output_shape,
        HabanaOperator::CalculateStrides(
            output_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());

    out.AddOutputTensor(metaData);
    auto shape_metaData = TensorMetaData(
        shape_tensor_shape,
        HabanaOperator::CalculateStrides(
            shape_tensor_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());
    out.AddShapeTensor(shape_metaData);
    return out;

  } else {
    SetGuid(get_guid_with_precision("non_zero_v2_fwd"sv, self.scalar_type()));
    InferOutputMetaRetType out;
    // (i) This output_describing_shape_tensor is created to be used by
    // "reshape" node within CGUID. This should be created within CGUID in
    // future. (ii) This shape tensor should not be created in as part of
    // accumulation (lazy_kernels) else relationship between input tensor and
    // shape tensor st = f(input) is not preserved in all cases (e.g. min, max
    // shape inference with Calculated or Local Historic policies). (iii)
    // Creating shape tensor in back-end kernel is ok for cases where shape
    // tensor is strictly a function of another input tensor(s) and not a scalar
    // value coming from framework.
    // (iv) Please consult with vgoel@habana.ai before removing or modifying
    // this shape_tensor.
    auto st_shape = compute_output_st_shape(self);

    auto shape_metaData1 = TensorMetaData(
        st_shape,
        HabanaOperator::CalculateStrides(
            st_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());
    out.AddShapeTensor(shape_metaData1);
    auto output_shape = compute_output_shape(self);
    std::vector<int64_t> shape_tensor_shape = {5};

    auto metaData = TensorMetaData(
        output_shape,
        HabanaOperator::CalculateStrides(
            output_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());

    out.AddOutputTensor(metaData);
    auto metaData2 = TensorMetaData(
        shape_tensor_shape,
        HabanaOperator::CalculateStrides(
            shape_tensor_shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format());
    out.AddOutputTensor(metaData2);
    return out;
  }
}

void NonZeroOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  HABANA_ASSERT(
      inputs.size() == 2,
      "Incorrect size of inputs expected for NonZero operator");
  HABANA_ASSERT(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for NonZero operator");
  HABANA_ASSERT(
      output_metadata.size() == 2,
      "output_metadata expected to be vector of size 2");

  auto self = inputs[0].toTensor();
  if (self.dim() > 4) {
    auto output_shape = compute_output_shape(self);
    auto shape_tensor_shape = DimVector{5};
    auto cordinates_of_true = habana::createPTTensor(
        self,
        output_shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        output_metadata.at(0).persistent);
    auto shape_tensor = habana::createPTTensor(
        self,
        shape_tensor_shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        output_metadata.at(1).persistent);
    // shape_tensor is of type UINT32 not supported by ScalarType, use
    // synDataType
    synDataType synType = syn_type_uint32;
    AllocateSynapseOutput(graph, cordinates_of_true, output_metadata.at(0));
    AllocateSynapseOutput(
        graph,
        shape_tensor,
        synType,
        output_metadata.at(1),
        graph.is_dynamic_graph());
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    SetGuid(get_guid_with_precision("non_zero_v2_fwd"sv, self.scalar_type()));

    // (i) This output_describing_shape_tensor is created to be used by
    // "reshape" node within CGUID. This should be created within CGUID in
    // future. (ii) This shape tensor should not be created in as part of
    // accumulation (lazy_kernels) else relationship between input tensor and
    // shape tensor st = f(input) is not preserved in all cases (e.g. min, max
    // shape inference with Calculated or Local Historic policies). (iii)
    // Creating shape tensor in back-end kernel is ok for cases where shape
    // tensor is strictly a function of another input tensor(s) and not a
    // scalar value coming from framework. (iv) Please consult with
    // vgoel@habana.ai before removing or modifying this shape_tensor.
    auto st_shape = compute_output_st_shape(self);
    Tensor reshape_shape_tensor = habana::createPTTensor(
        self, st_shape, self.options(), self.suggest_memory_format(), false);
    AllocateSynapseShapeTensor(graph, reshape_shape_tensor);

    auto output_shape = compute_output_shape(self);
    auto shape_tensor_shape = DimVector{5};
    auto cordinates_of_true = habana::createPTTensor(
        self,
        output_shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        output_metadata.at(0).persistent);
    auto shape_tensor = habana::createPTTensor(
        self,
        shape_tensor_shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Int,
        output_metadata.at(1).persistent);
    // shape_tensor is of type UINT32 not supported by ScalarType, use
    // synDataType
    synDataType synType = syn_type_uint32;
    AllocateSynapseOutput(graph, cordinates_of_true, output_metadata.at(0));
    AllocateSynapseOutput(
        graph, shape_tensor, synType, output_metadata.at(1), false);

    ns_NonzeroV2::Params params = {};
    params.group_size = 64;
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

static auto& NonZeroKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu::nonzero",
        habana::NonZeroOperator);
