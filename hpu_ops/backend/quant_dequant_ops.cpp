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

#include "hpu_ops/quant_dequant_ops.h"

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
      "quantize_per_tensor_v2 expects only double or int parameters");
  if (scalar.isDouble()) {
    scalar_tensors.emplace_back(
        op->BuildConstantTensor(op, graph, scalar.toDouble(), force_type));
  } else {
    scalar_tensors.emplace_back(
        op->BuildConstantTensor(op, graph, scalar.toInt(), force_type));
  }
  syn_inputs.push_back(scalar_tensors.back().get());
}

OutputMetaDataVector QuantizePerTensorMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype = stack[5].toScalarType();
  return {meta};
}

QuantizePerTensor::QuantizePerTensor(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "quantize_per_tensor",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(QuantizePerTensorMeta);
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
      get_guid_with_precision("quantize_per_tensor_v2", self.scalar_type()),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}});
  syn_out(0) = std::move(op[0]);
}

OutputMetaDataVector DequantizePerTensorMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype =
      stack[6].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);
  return {meta};
}

DequantizePerTensor::DequantizePerTensor(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "dequantize_per_tensor",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(DequantizePerTensorMeta);
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
      get_guid_with_precision("dequantize_per_tensor_v2", out_dtype),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}});
  syn_out(0) = std::move(op[0]);
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

OutputMetaDataVector QuantizePerChannelMeta(const at::Stack& stack) {
  OutputMetaDataVector meta(1);
  meta.at(0).shape = stack_tensor(stack, 0).sizes().vec();
  meta.at(0).dtype = stack[6].toScalarType();

  return meta;
}

QuantizePerChannel::QuantizePerChannel(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "quantize_per_channel",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetFillParams(FillQuantizePerChannelParams);
  SetOutputMetaFn(QuantizePerChannelMeta);
}

std::shared_ptr<void> FillDequantizePerChannelParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_QuantizationPerChannel::ParamsV2);
  params->axis = stack[3].toInt();
  return params;
}

OutputMetaDataVector DequantizePerChannelMeta(const at::Stack& stack) {
  OutputMetaDataVector meta(1);
  meta.at(0).shape = stack_tensor(stack, 0).sizes().vec();
  meta.at(0).dtype =
      stack[7].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);

  return meta;
}

DequantizePerChannel::DequantizePerChannel(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "dequantize_per_channel",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetFillParams(FillDequantizePerChannelParams);
  SetOutputMetaFn(DequantizePerChannelMeta);
}

void DequantizePerChannel::CustomHandler(sh::graph&, at::Stack& stack) {
  SetGuid(get_guid_with_precision(
      "dequantize_per_channel",
      stack[7].toOptional<at::ScalarType>().value_or(at::ScalarType::Float)));
}

} // namespace habana

static const auto& QuantizePerTensorKernelRegistry =
    habana::KernelRegistry()
        .add(
            "quantized_decomposed::quantize_per_tensor",
            KERNEL_FN_GLOBAL(habana::QuantizePerTensor))
        .add(
            "quantized_decomposed::quantize_per_tensor.tensor",
            KERNEL_FN_GLOBAL(habana::QuantizePerTensor))
        .add(
            "quantized_decomposed::quantize_per_tensor.tensor2",
            KERNEL_FN_GLOBAL(habana::QuantizePerTensor))
        .add(
            "quantized_decomposed::dequantize_per_tensor",
            KERNEL_FN_GLOBAL(habana::DequantizePerTensor))
        .add(
            "quantized_decomposed::dequantize_per_tensor.tensor",
            KERNEL_FN_GLOBAL(habana::DequantizePerTensor))
        .add(
            "quantized_decomposed::dequantize_per_tensor.tensor2",
            KERNEL_FN_GLOBAL(habana::DequantizePerTensor))
        .add(
            "quantized_decomposed::quantize_per_channel",
            KERNEL_FN_GLOBAL(habana::QuantizePerChannel))
        .add(
            "quantized_decomposed::dequantize_per_channel",
            KERNEL_FN_GLOBAL(habana::DequantizePerChannel));
