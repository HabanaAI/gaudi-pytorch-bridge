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

#include "hpu_ops/int4_ops.h"

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector ConvertFromInt4Meta(const at::Stack& stack) {
  auto output_shape = stack[0].toTensor().sizes().vec();
  output_shape.back() *= 8;
  OutputMetaDataVector meta(1);
  meta.at(0).shape = output_shape;
  meta.at(0).dtype = stack[3].toScalarType();

  return meta;
}

Int4BaseOp::Int4BaseOp(
    int device_id,
    c10::ScalarType scalar_type,
    const std::string& guid)
    : OpBackend(device_id, guid, scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(ConvertFromInt4Meta);
}

void Int4BaseOp::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto meta = ConvertFromInt4Meta(stack)[0];

  std::vector<synTensor> inputs{syn_in(0), syn_in(1)};
  if (stack[2].isTensor()) {
    inputs.push_back(syn_in(2));
  }

  auto result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("dequantize_4_bit", meta.dtype),
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}}});

  syn_out(0) = std::move(result[0]);
}

ConvertFromInt4::ConvertFromInt4(int device_id, c10::ScalarType scalar_type)
    : Int4BaseOp(device_id, scalar_type, "convert_from_int4") {}

ConvertFromUint4::ConvertFromUint4(int device_id, c10::ScalarType scalar_type)
    : Int4BaseOp(device_id, scalar_type, "convert_from_uint4") {}

} // namespace habana

static const auto& CastKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::convert_from_int4",
            KERNEL_FN_GLOBAL(habana::ConvertFromInt4))
        .add(
            "hpu::convert_from_uint4",
            KERNEL_FN_GLOBAL(habana::ConvertFromUint4));
