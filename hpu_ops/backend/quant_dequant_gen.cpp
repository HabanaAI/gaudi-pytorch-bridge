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

#include "generated/backend/dequantize_per_channel.h"
#include "generated/backend/dequantize_per_tensor.h"
#include "generated/backend/quantize_per_channel.h"
#include "generated/backend/quantize_per_tensor.h"

namespace sh = synapse_helpers;
namespace habana {

void WrapScalarAsTensor(
    habana::OpBackend* op,
    sh::graph& graph,
    const c10::IValue& scalar,
    std::vector<sh::tensor>& scalar_tensors,
    std::vector<synTensor>& syn_inputs,
    c10::ScalarType force_type) {
  TORCH_CHECK(
      scalar.isDouble() || scalar.isInt(),
      "quantize_per_tensor expects only double or int parameters");
  if (scalar.isDouble()) {
    scalar_tensors.emplace_back(
        op->BuildConstantTensor(op, graph, scalar.toDouble(), force_type));
  } else {
    scalar_tensors.emplace_back(
        op->BuildConstantTensor(op, graph, scalar.toInt(), force_type));
  }
  syn_inputs.push_back(scalar_tensors.back().get());
}

std::shared_ptr<void> FillQuantizePerChannelParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_QuantizationPerChannel::ParamsV2);
  params->axis = stack[3].toInt();
  params->quant_min = stack[4].toInt();
  params->quant_max = stack[5].toInt();
  return params;
}

std::shared_ptr<void> FillDequantizePerChannelParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_QuantizationPerChannel::ParamsV2);
  params->axis = stack[3].toInt();
  return params;
}

OutputMetaDataVector QuantizePerTensorMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype = stack[5].toScalarType();
  return {meta};
}

OutputMetaDataVector DequantizePerTensorMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype =
      stack[6].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);
  return {meta};
}

OutputMetaDataVector QuantizePerChannelMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype = stack[6].toScalarType();
  return {meta};
}

OutputMetaDataVector DequantizePerChannelMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype =
      stack[7].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);
  return {meta};
}

void QuantizePerTensor::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto scale = stack.at(1);
  auto zero_point = stack.at(2);
  auto quant_min = stack.at(3);
  auto quant_max = stack.at(4);

  auto force_type = self.scalar_type();
  std::vector<synTensor> syn_inputs{syn_in(0)};
  std::vector<sh::tensor> scalar_tensors;

  if (scale.isTensor()) {
    syn_inputs.push_back(syn_in(1));
    syn_inputs.push_back(syn_in(2));
    if (quant_min.isTensor()) {
      // quantize_per_tensor_tensor2
      syn_inputs.push_back(syn_in(3));
      syn_inputs.push_back(syn_in(4));
    } else {
      // quantize_per_tensor_tensor
      WrapScalarAsTensor(
          this, graph, quant_min, scalar_tensors, syn_inputs, force_type);
      WrapScalarAsTensor(
          this, graph, quant_max, scalar_tensors, syn_inputs, force_type);
    }
  } else {
    // quantize_per_tensor
    WrapScalarAsTensor(
        this, graph, scale, scalar_tensors, syn_inputs, force_type);
    WrapScalarAsTensor(
        this, graph, zero_point, scalar_tensors, syn_inputs, force_type);
    WrapScalarAsTensor(
        this, graph, quant_min, scalar_tensors, syn_inputs, force_type);
    WrapScalarAsTensor(
        this, graph, quant_max, scalar_tensors, syn_inputs, force_type);
  }

  const auto meta = QuantizePerTensorMeta(stack)[0];
  auto op = BuildOp(
      graph,
      get_guid_with_precision("quantize_per_tensor", self.scalar_type()),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}});
  syn_out(0) = std::move(op[0]);
}

void DequantizePerTensor::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto scale = stack.at(1);
  auto zero_point = stack.at(2);
  auto out_dtype =
      stack.at(6).toOptional<at::ScalarType>().value_or(at::ScalarType::Float);

  std::vector<synTensor> syn_inputs{syn_in(0)};
  std::vector<sh::tensor> scalar_tensors;

  if (scale.isTensor()) {
    // dequantize_per_tensor_tensor and dequantize_per_tensor_tensor2
    syn_inputs.push_back(syn_in(1));
    syn_inputs.push_back(syn_in(2));
  } else {
    // dequantize_per_tensor
    WrapScalarAsTensor(
        this, graph, scale, scalar_tensors, syn_inputs, out_dtype);
    WrapScalarAsTensor(
        this, graph, zero_point, scalar_tensors, syn_inputs, out_dtype);
  }

  const auto meta = DequantizePerTensorMeta(stack)[0];
  auto op = BuildOp(
      graph,
      get_guid_with_precision("dequantize_per_tensor", out_dtype),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}});
  syn_out(0) = std::move(op[0]);
}

void DequantizePerChannel::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto out_dtype =
      stack[7].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);

  std::vector<synTensor> syn_inputs = {syn_in(0), syn_in(1)};
  if (stack.at(2).toOptional<at::Tensor>().has_value()) {
    syn_inputs.push_back(syn_in(2));
  }

  size_t size = 0;
  const auto& params = FillQuantizePerChannelParams(stack, size);
  const auto meta = DequantizePerChannelMeta(stack)[0];

  auto op = BuildOp(
      graph,
      update_guid_dtype(guid_, out_dtype),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(op[0]);
}

} // namespace habana
