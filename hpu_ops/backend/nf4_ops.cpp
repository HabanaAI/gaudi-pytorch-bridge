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

#include "hpu_ops/nf4_ops.h"
#include "hpu_ops/hpu_op_helper.h"

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector DequantizeNF4Meta(const at::Stack& stack) {
  OutputMetaDataVector meta(1);
  meta.at(0).shape = stack[3].toIntList().vec();
  meta.at(0).dtype = stack[4].toScalarType();
  return meta;
}

OutputMetaDataVector QuantizeNF4Meta(const at::Stack& stack) {
  OutputMetaDataVector meta(2);
  auto num_elements = stack[0].toTensor().numel();
  auto block_size = stack[1].toInt();

  meta.at(0).shape = {(num_elements + 1) / 2};
  meta.at(0).dtype = at::ScalarType::Byte;
  meta.at(1).shape = {(num_elements + block_size - 1) / block_size};
  meta.at(1).dtype = stack[0].toTensor().scalar_type();
  return meta;
}

FillParamsT FillDequantizeNF4Params(const at::Stack& stack) {
  PARAMS_STUB(ns_CastNF4Kernel::ParamsV2);
  params->group_size = stack[2].toInt();
  if (stack[5].toBool()) {
    params->big_endian = true;
  } else {
    params->big_endian = false;
  }
  return paramsT;
}

FillParamsT FillQuantizeNF4Params(const at::Stack& stack) {
  PARAMS_STUB(ns_CastNF4Kernel::ParamsV2);
  params->group_size = stack[1].toInt();
  params->big_endian = true;
  return paramsT;
}

void DequantizeNF4::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto meta = DequantizeNF4Meta(stack)[0];
  auto sizes = stack[0].toTensor().sizes().vec();
  auto params = FillDequantizeNF4Params(stack);
  // Need to this cast because we are getting uint8 dtype
  // need to convert to packed_nf4 dtype
  auto cast_to_NF4 = OpBackend::BuildNode(
      this,
      graph,
      {"reinterpret_cast",
       {syn_in(0)},
       {{sizes,
         c10::ScalarType::Byte,
         std::nullopt,
         DATA_TENSOR,
         syn_type_packed_nf4}}});
  // change the guid dtype based on meta dtype
  guid_ = get_guid_with_precision(
      [] {
        using namespace std::literals;
        return "cast_packed_nf4_to"sv;
      }(),
      meta.dtype);

  std::vector<synTensor> inputs{cast_to_NF4[0].get(), syn_in(1)};
  auto result = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.ptr(),
       params.size()});

  syn_out(0) = std::move(result[0]);
}

void QuantizeNF4::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto meta = QuantizeNF4Meta(stack);
  auto params = FillQuantizeNF4Params(stack);

  // Need to change the guid name based on the dtype
  // of the input tensor
  guid_ = get_guid_with_precision(
      [] {
        using namespace std::literals;
        return "cast"sv;
      }(),
      meta[1].dtype);
  guid_ = guid_ + "_to_packed_nf4";

  std::vector<synTensor> inputs{syn_in(0)};
  auto result = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       std::move(inputs),
       {{meta[0].shape, meta[0].dtype, 0}, {meta[1].shape, meta[1].dtype, 1}},
       params.ptr(),
       params.size()});

  syn_out(0) = std::move(result[0]);
  syn_out(1) = std::move(result[1]);
}

DequantizeNF4::DequantizeNF4(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "cast_packed_nf4_to",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(DequantizeNF4Meta);
  SetFillParams(FillDequantizeNF4Params);
}

QuantizeNF4::QuantizeNF4(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "cast_packed_nf4_from",
          scalar_type,
          {0, 0},
          {},
          {},
          false) {
  SetOutputMetaFn(QuantizeNF4Meta);
  SetFillParams(FillQuantizeNF4Params);
}

} // namespace habana

static const auto& CastDequantKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu::dequantize_nf4",
        habana::DequantizeNF4);

static const auto& CastQuantKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu::quantize_nf4",
        habana::QuantizeNF4);
