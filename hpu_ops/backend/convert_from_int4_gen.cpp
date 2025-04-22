/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
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
  auto input = stack[0].toTensor();
  auto output_shape = input.sizes().vec();
  auto out_dtype = stack[3].toScalarType();
  auto group_index = stack[4].toOptional<at::Tensor>();
  output_shape.back() *= 8;
  OutputMetaDataVector meta(1);
  meta.at(0).shape = output_shape;
  meta.at(0).dtype = out_dtype;

  TORCH_CHECK(
      not group_index.has_value() || out_dtype == at::kBFloat16,
      "Only BFloat16 out_dtype is supported for dequantization with group index");
  TORCH_CHECK(
      input.scalar_type() == at::kInt, "Only int input (weights) is supported");

  return meta;
}

void ConvertFromInt4::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "ConvertFromInt4::AddNode");

  auto input = stackGetter.getNextInput<TensorsPair>();
  auto scale = stackGetter.getNextInput<TensorsPair>();
  auto zero_point = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto out_dtype = stackGetter.getNextInput<at::ScalarType>();
  auto group_index = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto disable_fp8_clipping = stackGetter.getNextInput<bool>();

  auto out_shape = input.pt_t.sizes().vec();
  out_shape.back() *= 8;

  auto syn_type =
      (guid_ == "convert_from_uint4_i32") ? syn_type_uint4 : syn_type_int4;
  at::ScalarType scalar_type = at::kInt;
  auto new_input = OpBackend::BuildNode(
      this,
      graph,
      {"reinterpret_cast",
       {input.syn_t},
       {{out_shape, scalar_type, std::nullopt, DATA_TENSOR, syn_type}}});
  std::vector<synTensor> inputs{new_input[0].get(), scale.syn_t};

  if (group_index) {
    inputs.push_back(group_index->syn_t);
  }
  std::vector<sh::tensor> zp_tensors;
  if (zero_point) {
    if (zero_point->pt_t.scalar_type() == at::kInt) {
      auto zero_point_sizes = zero_point->pt_t.sizes().vec();
      zero_point_sizes.back() *= 8;
      zp_tensors = OpBackend::BuildNode(
          this,
          graph,
          {"reinterpret_cast",
           {zero_point->syn_t},
           {{zero_point_sizes,
             scalar_type,
             std::nullopt,
             DATA_TENSOR,
             syn_type}}});
      inputs.push_back(zp_tensors[0].get());
    } else {
      inputs.push_back(zero_point->syn_t);
    }
  }

  using namespace std::literals;
  if (group_index && syn_type == syn_type_int4) {
    auto result = OpBackend::BuildNode(
        this,
        graph,
        {
            "int4_group_index_dequantize_i4",
            std::move(inputs),
            {{out_shape, out_dtype, 0}},
        });
    syn_out(0) = std::move(result[0]);
  } else {
    ns_CastKernel::ParamsV3 params{};
    if (out_dtype == at::ScalarType::Float8_e5m2 ||
        out_dtype == at::ScalarType::Float8_e4m3fn) {
      if (disable_fp8_clipping) {
        PT_BRIDGE_DEBUG("FP8 clipping in ", guid_, " op is disabled.");
      } else {
        params.mode = CAST_CLIP;
      }
    }
    auto result = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("dequantize_4_bit"sv, out_dtype),
         std::move(inputs),
         {{out_shape, out_dtype, 0}},
         &params,
         sizeof(params)});
    syn_out(0) = std::move(result[0]);
  }
}

} // namespace habana
