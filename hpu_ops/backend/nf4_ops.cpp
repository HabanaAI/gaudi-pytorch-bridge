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

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector DequantizeNF4Meta(const at::Stack& stack) {
  OutputMetaDataVector meta(1);
  meta.at(0).shape = stack[3].toIntList().vec();
  meta.at(0).dtype = stack[4].toScalarType();
  return meta;
}

std::shared_ptr<void> FillDequantizeNF4Params(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_CastNF4Kernel::Params);
  params->group_size = stack[2].toInt();
  return params;
}

void DequantizeNF4::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto meta = DequantizeNF4Meta(stack)[0];
  size_t size = 0;
  auto params = FillDequantizeNF4Params(stack, size);
  // change the guid dtype based on meta dtype
  guid_ = get_guid_with_precision(
      [] {
        using namespace std::literals;
        return "cast_packed_nf4_to"sv;
      }(),
      meta.dtype);

  std::vector<synTensor> inputs{syn_in(0), syn_in(1)};
  auto result = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size});

  syn_out(0) = std::move(result[0]);
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

} // namespace habana

static const auto& CastKernelRegistry = habana::KernelRegistry().add(
    "hpu::dequantize_nf4",
    KERNEL_FN_GLOBAL(habana::DequantizeNF4));
