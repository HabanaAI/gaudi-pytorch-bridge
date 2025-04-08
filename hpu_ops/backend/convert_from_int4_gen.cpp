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

#include "generated/backend/convert_from_int4.h"
#include "generated/backend/convert_from_uint4.h"

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

void ConvertFromInt4::AddNode(sh::graph& graph, const at::Stack& stack) {
  auto meta = ConvertFromInt4Meta(stack)[0];

  std::vector<synTensor> inputs{syn_in(0), syn_in(1)};
  if (stack[2].isTensor()) {
    inputs.push_back(syn_in(2));
  }

  ns_CastKernel::ParamsV3 params{};
  if (meta.dtype == at::ScalarType::Float8_e5m2 or
      meta.dtype == at::ScalarType::Float8_e4m3fn) {
    const auto disable_fp8_clip = stack[4].toBool();
    if (disable_fp8_clip) {
      PT_BRIDGE_DEBUG("FP8 clipping in ", guid_, " op is disabled.");
    } else {
      params.mode = CAST_CLIP;
    }
  }

  using namespace std::literals;
  auto result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("dequantize_4_bit"sv, meta.dtype),
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(result[0]);
}

} // namespace habana
