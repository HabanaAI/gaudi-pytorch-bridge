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

namespace habana {

std::shared_ptr<void> FillQuantizePerTensorParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_QuantizationPerTensor::ParamsV2);
  if (stack[1].isDouble()) {
    params->scale = stack[1].toDouble();
    params->zero_point = stack[2].toInt();
  }
  if (stack[3].isInt()) {
    params->quant_min = stack[3].toInt();
    params->quant_max = stack[4].toInt();
  }
  return params;
}

OutputMetaDataVector QuantizePerTensorMeta(const at::Stack& stack) {
  OutputMetaDataVector meta(1);
  meta.at(0).shape = stack_tensor(stack, 0).sizes().vec();
  meta.at(0).dtype = stack[5].toScalarType();

  return meta;
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
  SetFillParams(FillQuantizePerTensorParams);
  SetOutputMetaFn(QuantizePerTensorMeta);
}

std::shared_ptr<void> FillDequantizePerTensorParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_QuantizationPerTensor::ParamsV2);
  if (stack[1].isDouble()) {
    params->scale = stack[1].toDouble();
    params->zero_point = stack[2].toInt();
  }
  return params;
}

OutputMetaDataVector DequantizePerTensorMeta(const at::Stack& stack) {
  OutputMetaDataVector meta(1);
  meta.at(0).shape = stack_tensor(stack, 0).sizes().vec();
  meta.at(0).dtype =
      stack[6].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);

  return meta;
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
  SetFillParams(FillDequantizePerTensorParams);
  SetOutputMetaFn(DequantizePerTensorMeta);
}

void DequantizePerTensor::CustomHandler(
    synapse_helpers::graph&,
    at::Stack& stack) {
  SetGuid(get_guid_with_precision(
      "dequantize_per_tensor",
      stack[6].toOptional<at::ScalarType>().value_or(at::ScalarType::Float)));
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

void DequantizePerChannel::CustomHandler(
    synapse_helpers::graph&,
    at::Stack& stack) {
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
