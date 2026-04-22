/**
 * Copyright (c) 2021-2026 Intel Corporation
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
#include "habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace sh = synapse_helpers;
namespace habana {
using namespace std::string_view_literals;

void WrapScalarAsTensor(
    habana::OpBackend* op,
    sh::graph& graph,
    const c10::IValue& scalar,
    std::vector<sh::tensor>& scalar_tensors,
    std::vector<synTensor>& syn_inputs,
    c10::ScalarType force_type) {
  HABANA_ASSERT(
      scalar.isDouble() || scalar.isInt(),
      "quantize_per_tensor expects only double or int parameters");
  if (scalar.isDouble()) {
    scalar_tensors.emplace_back(
        OpBackend::BuildConstantTensor(
            op, graph, scalar.toDouble(), force_type));
  } else {
    scalar_tensors.emplace_back(
        OpBackend::BuildConstantTensor(op, graph, scalar.toInt(), force_type));
  }
  syn_inputs.push_back(scalar_tensors.back().get());
}

FillParamsT FillQuantizePerChannelParams(const at::Stack& stack) {
  PARAMS_STUB(ns_QuantizationPerChannel::ParamsV2);
  params->axis = safe_convert<int>(stack[3].toInt());
  params->quant_min = safe_convert<int>(stack[4].toInt());
  params->quant_max = safe_convert<int>(stack[5].toInt());
  return paramsT;
}

FillParamsT FillDequantizePerChannelParams(const at::Stack& stack) {
  PARAMS_STUB(ns_QuantizationPerChannel::ParamsV2);
  params->axis = safe_convert<int>(stack[3].toInt());
  return paramsT;
}

OutputMetaDataVector QuantizePerTensorMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype = stack[5].toScalarType();
  return metaVec;
}

OutputMetaDataVector DequantizePerTensorMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype =
      stack[6].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);
  return metaVec;
}

OutputMetaDataVector QuantizePerChannelMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype = stack[6].toScalarType();
  return metaVec;
}

OutputMetaDataVector DequantizePerChannelMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype =
      stack[7].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);
  return metaVec;
}

using namespace std::literals;

void QuantizePerTensor::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  const auto& scale = stack.at(1);
  const auto& zero_point = stack.at(2);
  const auto& quant_min = stack.at(3);
  const auto& quant_max = stack.at(4);

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
      get_guid_with_precision("quantize_per_tensor"sv, self.scalar_type()),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}});
  syn_out(0) = std::move(op[0]);
}

void DequantizePerTensor::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  const auto& scale = stack.at(1);
  const auto& zero_point = stack.at(2);
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
      get_guid_with_precision("dequantize_per_tensor"sv, out_dtype),
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

  const auto& params = FillQuantizePerChannelParams(stack);
  const auto meta = DequantizePerChannelMeta(stack)[0];

  auto op = BuildOp(
      graph,
      update_guid_dtype(guid_, out_dtype),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}},
      params.ptr(),
      params.size());
  syn_out(0) = std::move(op[0]);
}

} // namespace habana
