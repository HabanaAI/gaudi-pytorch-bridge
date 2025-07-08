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

std::vector<int64_t> GetUnpackedShape(const at::Tensor& tensor) {
  auto shape_vec = tensor.sizes().vec();
  if (!shape_vec.empty()) {
    shape_vec.back() *= 8;
  }
  return shape_vec;
}

void ValidateShapeGrouped(
    const at::Tensor& input,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& zero_point,
    const at::Tensor& group_index,
    const at::ScalarType out_dtype) {
  auto unpacked_input_shape = GetUnpackedShape(input);

  TORCH_CHECK(
      input.dim() == 2 && scale.dim() == 2,
      "Grouped dequantization requires both input and scale to be 2D tensors. Got input shape: ",
      input.sizes(),
      ", scale shape: ",
      scale.sizes());

  TORCH_CHECK(
      input.size(0) % scale.size(0) == 0,
      "The number of rows in the input (",
      input.size(0),
      ") must be divisible by the number of rows in the scale (",
      scale.size(0),
      "). Got input shape: ",
      input.sizes(),
      ", scale shape: ",
      scale.sizes());

  TORCH_CHECK(
      unpacked_input_shape[1] == scale.size(1),
      "The number of columns in the output (",
      unpacked_input_shape[1],
      ") must match the number of columns in the scale (",
      scale.size(1),
      "). Got output shape: ",
      unpacked_input_shape,
      ", scale shape: ",
      scale.sizes());

  if (zero_point.has_value()) {
    auto unpacked_zero_point_shape = GetUnpackedShape(*zero_point);
    TORCH_CHECK(
        unpacked_zero_point_shape == scale.sizes(),
        "The unpacked zero_point shape (",
        unpacked_zero_point_shape,
        ") must match the scale shape (",
        scale.sizes(),
        ").");
  }

  TORCH_CHECK(
      group_index.dim() == 1 && group_index.size(0) == unpacked_input_shape[0],
      "The group_index must be a 1D tensor, and its length (",
      group_index.size(0),
      ") must match the number of rows in the output (",
      unpacked_input_shape[0],
      "). Got group_index shape: ",
      group_index.sizes(),
      ", output shape: ",
      unpacked_input_shape);

  TORCH_CHECK(
      out_dtype == at::kBFloat16,
      "Only BFloat16 output dtype is supported for grouped dequantization with group index.");
}

bool is_equal_except_one_and_divisible(
    const at::IntArrayRef& a,
    const at::IntArrayRef& b) {
  if (a.size() != b.size())
    return false;
  size_t diff_idx = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      if (diff_idx != std::numeric_limits<size_t>::max())
        return false; // More than one dim differs
      diff_idx = i;
    }
  }
  if (diff_idx == std::numeric_limits<size_t>::max()) {
    return false; // All dims equal, must differ in one
  }
  return a[diff_idx] % b[diff_idx] == 0;
}

void ValidateShapeNonGrouped(
    const at::Tensor& input,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& zero_point) {
  auto unpacked_shape = GetUnpackedShape(input);

  TORCH_CHECK(
      input.dim() >= 1 && input.dim() <= 4,
      "The input tensor must have 1 to 4 dimensions. Got input with ",
      input.dim(),
      " dimensions.");

  TORCH_CHECK(
      scale.sizes() == unpacked_shape ||
          is_equal_except_one_and_divisible(unpacked_shape, scale.sizes()),
      "The scale tensor shape must match the output shape or be equal in all dimensions except one, which must divide the output dimension. Got scale shape: ",
      scale.sizes(),
      ", output shape: ",
      unpacked_shape);

  if (zero_point.has_value()) {
    auto zp_shape = zero_point->sizes().back() == input.sizes().back()
        ? GetUnpackedShape(*zero_point)
        : zero_point->sizes().vec();
    TORCH_CHECK(
        zp_shape == unpacked_shape ||
            is_equal_except_one_and_divisible(unpacked_shape, zp_shape),
        "The zero_point tensor shape must match the packed input shape, unpacked output shape, or be equal in all dimensions except one, which must divide the output dimension. Got zero_point shape: ",
        zp_shape,
        ", output shape: ",
        unpacked_shape);

    auto zp_dtype = zero_point.value().scalar_type();
    TORCH_CHECK(
        zp_dtype == input.scalar_type() || zp_dtype == scale.scalar_type(),
        "The zero_point dtype (",
        zp_dtype,
        ") must match the input dtype (",
        input.scalar_type(),
        ") or the scale dtype (",
        scale.scalar_type(),
        ").");
  }
}

OutputMetaDataVector ConvertFromInt4MetaCommon(
    const at::Tensor& input,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& zero_point,
    const at::ScalarType out_dtype,
    const std::optional<at::Tensor>& group_index,
    const bool disable_fp8_clipping) {
  TORCH_CHECK(
      !(group_index.has_value() && disable_fp8_clipping),
      "The group_index and disable_fp8_clipping options cannot be used together.");

  TORCH_CHECK(
      input.scalar_type() == at::kInt, "Only int input tensors are supported");

  if (group_index.has_value()) {
    ValidateShapeGrouped(
        input, scale, zero_point, group_index.value(), out_dtype);
  } else {
    ValidateShapeNonGrouped(input, scale, zero_point);
  }

  OutputMetaDataVector meta(1);
  meta.at(0).shape = GetUnpackedShape(input);
  meta.at(0).dtype = out_dtype;

  return meta;
}

OutputMetaDataVector ConvertFromInt4Meta(const at::Stack& stack) {
  auto input = stack[0].toTensor();
  auto scale = stack[1].toTensor();
  auto zero_point = stack[2].toOptional<at::Tensor>();
  auto out_dtype = stack[3].toScalarType();
  auto group_index = stack[4].toOptional<at::Tensor>();
  auto disable_fp8_clipping = stack[5].toBool();
  return ConvertFromInt4MetaCommon(
      input, scale, zero_point, out_dtype, group_index, disable_fp8_clipping);
}

void ConvertFromInt4::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "ConvertFromInt4::AddNode");

  auto input = stackGetter.getNextInput<TensorsPair>();
  auto scale = stackGetter.getNextInput<TensorsPair>();
  auto zero_point = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto out_dtype = stackGetter.getNextInput<at::ScalarType>();
  auto group_index = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto disable_fp8_clipping = stackGetter.getNextInput<bool>();

  synDataType syn_type;
  if (guid_ == "convert_from_int4_i32")
    syn_type = syn_type_int4;
  else if (guid_ == "convert_from_uint4_i32") {
    syn_type = syn_type_uint4;
  } else
    AT_ERROR("Unexpected guid: " + guid_);

  auto ReinterpretCast = [&](const TensorsPair& tensor) {
    auto unpacked_shape = tensor.pt_t.sizes().vec();
    unpacked_shape.back() *= 8;
    return OpBackend::BuildNode(
        this,
        graph,
        {"reinterpret_cast",
         {tensor.syn_t},
         {{unpacked_shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type}}});
  };

  auto unpacked_input = ReinterpretCast(input);
  std::vector<synTensor> inputs{unpacked_input[0].get(), scale.syn_t};

  if (group_index) {
    inputs.push_back(group_index->syn_t);
  }
  std::vector<sh::tensor> unpacked_zp;
  if (zero_point) {
    auto zp_packed =
        zero_point->pt_t.sizes().back() == input.pt_t.sizes().back();
    if (zp_packed) {
      unpacked_zp = ReinterpretCast(*zero_point);
      inputs.push_back(unpacked_zp[0].get());
    } else {
      inputs.push_back(zero_point->syn_t);
    }
  }

  auto out_shape = input.pt_t.sizes().vec();
  out_shape.back() *= 8;
  std::vector<sh::tensor> result;
  NodeAttr node_attr;
  if (group_index) {
    node_attr = {
        syn_type == syn_type_int4 ? "grouped_index_transpose_dequant_fwd_i4"
                                  : "grouped_index_transpose_dequant_fwd_u4",
        std::move(inputs),
        {{out_shape, out_dtype, 0}},
    };
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
    node_attr = {
        get_guid_with_precision("dequantize_4_bit"sv, out_dtype),
        std::move(inputs),
        {{out_shape, out_dtype, 0}},
        &params,
        sizeof(params)};
  }
  result = OpBackend::BuildNode(this, graph, std::move(node_attr));
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
