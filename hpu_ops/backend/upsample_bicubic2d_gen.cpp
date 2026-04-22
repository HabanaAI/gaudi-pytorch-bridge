/**
 * Copyright (c) 2025-2026 Intel Corporation
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

#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/upsample_bicubic2d.h"
#include "hpu_ops/custom_op_outshape.h"
#include "upsample_utils.h"

namespace habana {

using namespace upsample_utils;
using namespace synapse_helpers::layouts;

// -------> Forward <-------

FillParamsT FillUpsampleBicubic2DFwdParams(const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto align_corners = stack.at(2).toBool();
  const auto& scales = stack.at(3);
  const bool antialias = false;

  auto getScales = [&]() -> std::optional<std::array<double, 3>> {
    if (not scales.isNone()) {
      if (not scales.isScalar()) {
        const auto scales_vec = scales.toDoubleVector();
        return std::array<double, 3>{1.0, scales_vec.at(0), scales_vec.at(1)};
      }
      return std::array<double, 3>{
          1.0, stack.at(3).toDouble(), stack.at(4).toDouble()};
    }
    return std::nullopt;
  };

  const auto output_size_opt = (not stack.at(1).isNone())
      ? std::make_optional(stack.at(1).toIntVector())
      : std::nullopt;
  const auto scales_opt = getScales();

  return FillResizeParams(
      bicubic, output_size_opt, scales_opt, align_corners, antialias);
}

namespace {

template <class InputSizeT>
sizes_vec_template<InputSizeT> upsample_bicubic2d_output_shape_impl(
    at::ArrayRef<InputSizeT> input_size,
    at::ArrayRef<int64_t> output_size) {
  std::vector<InputSizeT> output_shape{
      input_size[INPUT_N_IDX], input_size[INPUT_C_IDX]};

  output_shape.insert(
      output_shape.end(), output_size.begin(), output_size.end());

  return {output_shape};
}

template <class InputSizeT, class OutputSizeT>
sizes_vec_template<InputSizeT> upsample_bicubic2d_output_shape_impl(
    at::ArrayRef<InputSizeT> input_size,
    at::ArrayRef<OutputSizeT> output_size) {
  std::vector<InputSizeT> output_shape{
      input_size[INPUT_N_IDX], input_size[INPUT_C_IDX]};

  output_shape.push_back(
      input_size[INPUT_H_IDX] * static_cast<InputSizeT>(output_size[0]));
  output_shape.push_back(
      input_size[INPUT_W_IDX] * static_cast<InputSizeT>(output_size[1]));

  return {output_shape};
}

} // namespace

sym_sizes_vec upsample_bicubic2d_vec_output_shape(
    const std::vector<at::Tensor>& inputs,
    const std::optional<std::vector<int64_t>>& ints,
    const std::optional<std::vector<float>>& floats) {
  HABANA_ASSERT(inputs.size() == 1);
  HABANA_ASSERT(
      (ints and ints->size() == 2) or (floats and floats->size() == 2));

  bool is_output_size_provided = ints.has_value();
  return (is_output_size_provided)
      ? upsample_bicubic2d_output_shape_impl(
            inputs[0].sym_sizes(), at::ArrayRef<int64_t>(*ints))
      : upsample_bicubic2d_output_shape_impl(
            inputs[0].sym_sizes(), at::ArrayRef<float>(*floats));
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(
    upsample_bicubic2d_vec,
    upsample_bicubic2d_vec_output_shape);

OutputMetaDataVector UpsampleBicubic2DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  const auto& output_size = stack.at(1);
  const auto& scales = stack.at(3);

  upsample_2d_common_check(self, output_size, scales);
  check_null_input(output_size, scales);

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();

  if (not output_size.isNone()) {
    meta.shape = upsample_bicubic2d_output_shape_impl(
        self.sizes(), at::ArrayRef<int64_t>(output_size.toIntVector()))[0];
  } else {
    meta.shape = upsample_bicubic2d_output_shape_impl(
        self.sizes(), at::ArrayRef<double>(scales.toDoubleVector()))[0];
  }
  meta.dtype = self.scalar_type();

  check_input_output_height_width(
      self.sizes()[2], meta.shape.at(2), self.sizes()[3], meta.shape.at(3));

  return metaVec;
}

// -------> Backward <-------

FillParamsT FillUpsampleBicubic2DBwdParams(const at::Stack& stack) {
  const auto grad_output = stack.at(0).toTensor();
  const auto& output_size = stack.at(1);
  const auto align_corners = stack.at(3).toBool();
  const auto& scales = stack.at(4);
  const bool antialias = false;

  const auto scale_d = 1.0;
  const auto scale_h = stack.at(4).toOptional<double>().value_or(1.0F);
  const auto scale_w = stack.at(5).toOptional<double>().value_or(1.0F);

  return FillResizeParams(
      grad_output.dim(),
      bicubic,
      output_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

OutputMetaDataVector UpsampleBicubic2DBwdMeta(const at::Stack& stack) {
  const auto grad_output = stack.at(0).toTensor();
  const auto& output_size = stack.at(1);
  const auto& scales = stack.at(4);

  check_null_input(output_size, scales);
  upsample_2d_common_check(grad_output, output_size, scales);

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();

  meta.shape = stack.at(2).toIntVector();
  meta.dtype = grad_output.scalar_type();

  return metaVec;
}

} // namespace habana
