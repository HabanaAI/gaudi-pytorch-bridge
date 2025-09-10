/**
 * Copyright (c) 2025 Intel Corporation
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

#include "generated/backend/upsample_trilinear3d.h"
#include "generated/backend/upsample_trilinear3d_backward.h"
#include "hpu_ops/custom_op_outshape.h"
#include "upsample_utils.h"

namespace habana {

using namespace upsample_utils;
using namespace synapse_helpers::layouts;

// -------> Forward <-------

namespace {
std::optional<std::array<double, 3>> getScales(const at::Stack& stack) {
  constexpr auto scales_idx = 3;
  const auto scales = stack.at(scales_idx);

  if (not scales.isNone()) {
    if (not scales.isScalar()) {
      const auto scales_vec = scales.toDoubleVector();
      return std::array<double, 3>{
          scales_vec.at(0), scales_vec.at(1), scales_vec.at(2)};
    }
    return std::array<double, 3>{
        stack.at(3).toDouble(), stack.at(4).toDouble(), stack.at(5).toDouble()};
  }
  return std::nullopt;
};
} // namespace

FillParamsT FillUpsampleTrilinear3DFwdParams(const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto align_corners = stack.at(2).toBool();
  const bool antialias = false;

  const auto output_size_opt = (not stack.at(1).isNone())
      ? std::make_optional(stack.at(1).toIntVector())
      : std::nullopt;
  const auto scales_opt = getScales(stack);

  return FillResizeParams(
      linear, output_size_opt, scales_opt, align_corners, antialias);
}

namespace {

template <class InputSizeT>
sizes_vec_template<InputSizeT> upsample_trilinear3d_output_shape_impl(
    at::ArrayRef<InputSizeT> input_size,
    at::ArrayRef<int64_t> output_size) {
  std::vector<InputSizeT> output_shape{
      input_size[INPUT_3D_N_IDX], input_size[INPUT_3D_C_IDX]};

  output_shape.insert(
      output_shape.end(), output_size.begin(), output_size.end());

  return {output_shape};
}

template <class InputSizeT, class OutputSizeT>
sizes_vec_template<InputSizeT> upsample_trilinear3d_output_shape_impl(
    at::ArrayRef<InputSizeT> input_size,
    at::ArrayRef<OutputSizeT> output_size) {
  std::vector<InputSizeT> output_shape{
      input_size[INPUT_3D_N_IDX], input_size[INPUT_3D_C_IDX]};

  output_shape.push_back(
      input_size[INPUT_3D_D_IDX] * static_cast<InputSizeT>(output_size[0]));
  output_shape.push_back(
      input_size[INPUT_3D_H_IDX] * static_cast<InputSizeT>(output_size[1]));
  output_shape.push_back(
      input_size[INPUT_3D_W_IDX] * static_cast<InputSizeT>(output_size[2]));

  return {output_shape};
}

} // namespace

sym_sizes_vec upsample_trilinear3d_vec_output_shape(
    const std::vector<at::Tensor>& inputs,
    const std::optional<std::vector<int64_t>>& ints,
    const std::optional<std::vector<float>>& floats) {
  HABANA_ASSERT(inputs.size() == 1);
  HABANA_ASSERT(
      (ints and ints->size() == 3) or (floats and floats->size() == 3));

  bool is_output_size_provided = ints.has_value();
  return (is_output_size_provided)
      ? upsample_trilinear3d_output_shape_impl(
            inputs[0].sym_sizes(), at::ArrayRef<int64_t>(*ints))
      : upsample_trilinear3d_output_shape_impl(
            inputs[0].sym_sizes(), at::ArrayRef<float>(*floats));
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(
    upsample_trilinear3d_vec,
    upsample_trilinear3d_vec_output_shape);

OutputMetaDataVector UpsampleTrilinear3DFwdMeta(const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto output_size = stack.at(1);
  const auto scales = stack.at(3);

  upsample_3d_common_check(self, output_size, scales);
  check_null_input(output_size, scales);

  OutputMetaData meta{};

  meta.shape =
      (not output_size.isNone()
           ? upsample_trilinear3d_output_shape_impl(
                 self.sizes(), at::ArrayRef<int64_t>(output_size.toIntVector()))
           : upsample_trilinear3d_output_shape_impl(
                 self.sizes(),
                 at::ArrayRef<double>(scales.toDoubleVector())))[0];
  meta.dtype = self.scalar_type();

  check_input_output_depth_height_width(
      self.sizes()[2],
      meta.shape.at(2),
      self.sizes()[3],
      meta.shape.at(3),
      self.sizes()[4],
      meta.shape.at(4));

  return {meta};
}

SharedMetaDataVector UpsampleTrilinear3DFwdSharedMeta(
    const at::Stack& stack,
    [[maybe_unused]] habana_helpers::HabanaExecutionMode exec_mode) {
  const auto align_corners = stack.at(2).toBool();
  constexpr auto scales_idx = 3;
  constexpr auto is_forward = true;

  return UpsampleCommmonSharedLayer(
      stack, align_corners, scales_idx, is_forward);
}

void UpsampleTrilinear3DFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  constexpr auto isForward = true;

  auto meta = UpsampleTrilinear3DFwdMeta(stack)[0];

  const auto self = stack.at(0).toTensor();
  const auto output_size = stack.at(1);
  const auto align_corners = stack.at(2).toBool();
  const auto scales = stack.at(3);

  const auto scales_opt = getScales(stack);
  const auto [scale_d, scale_h, scale_w] =
      scales_opt ? *scales_opt : std::array<double, 3>{1.0F, 1.0F, 1.0F};

  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      linear,
      isForward,
      {syn_in(0)},
      output_size,
      align_corners,
      scales,
      {scale_d, scale_h, scale_w},
      meta,
      self);
}

// -------> Backward <-------

OutputMetaDataVector UpsampleTrilinear3DBwdMeta(const at::Stack& stack) {
  const auto grad_output = stack.at(0).toTensor();
  const auto output_size = stack.at(1);
  const auto scales = stack.at(4);

  check_null_input(output_size, scales);
  upsample_3d_common_check(grad_output, output_size, scales);

  OutputMetaData meta{};

  meta.shape = stack.at(2).toIntVector();
  meta.dtype = grad_output.scalar_type();

  return {meta};
}

SharedMetaDataVector UpsampleTrilinear3DBwdSharedMeta(
    const at::Stack& stack,
    [[maybe_unused]] habana_helpers::HabanaExecutionMode exec_mode) {
  const auto align_corners = stack.at(3).toBool();
  constexpr auto scales_idx = 4;
  constexpr auto is_forward = false;

  return UpsampleCommmonSharedLayer(
      stack, align_corners, scales_idx, is_forward);
}

void UpsampleTrilinear3DBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  constexpr auto is_forward = false;

  auto meta = UpsampleTrilinear3DBwdMeta(stack)[0];

  const auto grad_output = stack.at(0).toTensor();
  const auto output_size = stack.at(1);
  const auto align_corners = stack.at(3).toBool();
  const auto scales = stack.at(4);

  const auto scale_d = stack.at(4).toOptional<double>().value_or(1.0F);
  const auto scale_h = stack.at(5).toOptional<double>().value_or(1.0F);
  const auto scale_w = stack.at(6).toOptional<double>().value_or(1.0F);

  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      linear,
      is_forward,
      {syn_in(0)},
      output_size,
      align_corners,
      scales,
      {scale_d, scale_h, scale_w},
      meta,
      grad_output);
}

} // namespace habana
