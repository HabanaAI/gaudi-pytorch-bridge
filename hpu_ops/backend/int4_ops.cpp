/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
