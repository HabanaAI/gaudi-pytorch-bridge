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

#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/_upsample_nearest_exact1d.h"
#include "generated/backend/_upsample_nearest_exact1d_backward.h"
#include "generated/backend/_upsample_nearest_exact2d.h"
#include "generated/backend/_upsample_nearest_exact2d_backward.h"
#include "generated/backend/_upsample_nearest_exact3d.h"
#include "generated/backend/_upsample_nearest_exact3d_backward.h"
#include "generated/backend/upsample_linear1d.h"
#include "generated/backend/upsample_linear1d_backward.h"
#include "generated/backend/upsample_nearest1d.h"
#include "generated/backend/upsample_nearest1d_backward.h"
#include "generated/backend/upsample_nearest2d.h"
#include "generated/backend/upsample_nearest2d_backward.h"
#include "generated/backend/upsample_nearest3d.h"
#include "generated/backend/upsample_nearest3d_backward.h"
#include "upsample_utils.h"

using namespace synapse_helpers::layouts;

namespace habana {

using namespace upsample_utils;

// Forward Meta Function - Linear1D
OutputMetaDataVector UpsampleLinear1DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  upsample_1d_common_check(self, out_size, scale);
  check_null_input(out_size, scale);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  if (!out_size.isNone()) {
    meta.shape = {
        self.sizes()[0], self.sizes()[1], out_size.toIntVector().at(0)};
  } else if (!scale.isNone() && !scale.isScalar()) {
    double scale_factor = scale.toDoubleVector().at(0);
    auto width = static_cast<double>(self.sizes()[2]);
    meta.shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(width * scale_factor)};
  }
  check_input_output_width(self.sizes()[2], meta.shape.at(2));
  return {meta};
}
// Backward Meta Function - Linear1D
OutputMetaDataVector UpsampleLinear1DBwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(4);
  upsample_1d_common_check(self, out_size, scale);
  OutputMetaData meta;
  meta.shape = stack.at(2).toIntVector();
  meta.dtype = self.scalar_type();
  return {meta};
}
// Forward Meta Function - Nearest1D
OutputMetaDataVector UpsampleNearest1DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(2);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  upsample_1d_common_check(self, out_size, scale);
  check_null_input(out_size, scale);
  if (!out_size.isNone()) {
    meta.shape = {
        self.sizes()[0], self.sizes()[1], out_size.toIntVector().at(0)};
  } else if (!scale.isNone()) {
    double scale_factor = scale.toDoubleVector().at(0);
    auto width = static_cast<double>(self.sizes()[2]);
    meta.shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(width * scale_factor)};
  }
  check_input_output_width(self.sizes()[2], meta.shape.at(2));
  return {meta};
}
// Backward Meta Function - Nearest1D
OutputMetaDataVector UpsampleNearest1DBwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  upsample_1d_common_check(self, out_size, scale);
  OutputMetaData meta;
  meta.shape = stack.at(2).toIntVector();
  meta.dtype = self.scalar_type();
  return {meta};
}
// Forward Output Shape - Bilinear2D
std::vector<int64_t> UpsampleBilinear2DFwdOutputShapeSynapseLayout(
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  std::vector<int64_t> out_shape;
  if (!out_size.isNone()) {
    out_shape = {
        self.sizes()[INPUT_N_IDX],
        self.sizes()[INPUT_C_IDX],
        out_size.toIntVector().at(0),
        out_size.toIntVector().at(1)};
  } else if (!scale.isNone()) {
    double scale_w = scale.toDoubleVector().at(1);
    double scale_h = scale.toDoubleVector().at(0);
    out_shape = {
        self.sizes()[INPUT_N_IDX],
        self.sizes()[INPUT_C_IDX],
        static_cast<int64_t>(
            static_cast<double>(self.sizes()[INPUT_H_IDX]) * scale_h),
        static_cast<int64_t>(
            static_cast<double>(self.sizes()[INPUT_W_IDX]) * scale_w)};
  }
  return out_shape;
}
// Forward Meta Function - Bilinear2D
OutputMetaDataVector UpsampleBilinear2DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  std::vector<int64_t> out_shape;
  upsample_2d_common_check(self, out_size, scale);
  check_null_input(out_size, scale);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = UpsampleBilinear2DFwdOutputShapeSynapseLayout(stack);

  check_input_output_height_width(
      self.sizes()[2], meta.shape.at(2), self.sizes()[3], meta.shape.at(3));
  return {meta};
}
// Backward Meta Function - Bilinear2D
OutputMetaDataVector UpsampleBilinear2DBwdMeta(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(4);
  check_null_input(out_size, scale);
  upsample_2d_common_check(grad_in, out_size, scale);

  OutputMetaData meta;
  meta.shape = stack.at(2).toIntVector();
  meta.dtype = grad_in.scalar_type();
  return {meta};
}
std::vector<int64_t> UpsampleNearest2DFwdOutputShapeSynapseLayout(
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(2);
  std::vector<int64_t> out_shape;
  if (!out_size.isNone()) {
    // NCHW
    out_shape = {
        self.sizes()[INPUT_N_IDX],
        self.sizes()[INPUT_C_IDX],
        out_size.toIntVector().at(0),
        out_size.toIntVector().at(1)};
  } else if (!scale.isNone()) {
    double scale_w = scale.toDoubleVector().at(1);
    double scale_h = scale.toDoubleVector().at(0);
    out_shape = {
        self.sizes()[INPUT_N_IDX],
        self.sizes()[INPUT_C_IDX],
        static_cast<int64_t>(
            static_cast<double>(self.sizes()[INPUT_H_IDX]) * scale_h),
        static_cast<int64_t>(
            static_cast<double>(self.sizes()[INPUT_W_IDX]) * scale_w)};
  }
  return out_shape;
}
std::vector<int64_t> UpsampleNearestExact2DFwdOutputShapeSynapseLayout(
    const at::Stack& stack) {
  auto self_sizes = stack.at(0).toTensor().sizes();
  auto out_size = stack.at(1).toIntVector();
  auto scale_h = stack.at(2).toOptional<double>().value_or(1.0);
  auto scale_w = stack.at(3).toOptional<double>().value_or(1.0);
  std::vector<int64_t> out_shape;
  if (!out_size.empty()) {
    // NCHW
    out_shape = {
        self_sizes.at(INPUT_N_IDX),
        self_sizes.at(INPUT_C_IDX),
        out_size.at(0),
        out_size.at(1)};
  } else if (scale_h != 1.0 || scale_w != 1.0) {
    out_shape = {
        self_sizes.at(INPUT_N_IDX),
        self_sizes.at(INPUT_C_IDX),
        static_cast<int64_t>(
            static_cast<double>(self_sizes.at(INPUT_H_IDX)) * scale_h),
        static_cast<int64_t>(
            static_cast<double>(self_sizes.at(INPUT_W_IDX)) * scale_w)};
  }
  return out_shape;
}
std::vector<int64_t> UpsampleNearestExact3DFwdOutputShapeSynapseLayout(
    const at::Stack& stack) {
  auto self_sizes = stack.at(0).toTensor().sizes();
  auto out_size = stack.at(1).toIntVector();
  auto scale_d = stack.at(2).toOptional<double>().value_or(1.0);
  auto scale_h = stack.at(3).toOptional<double>().value_or(1.0);
  auto scale_w = stack.at(4).toOptional<double>().value_or(1.0);
  std::vector<int64_t> out_shape;
  if (!out_size.empty()) {
    // NCDHW
    out_shape = {
        self_sizes.at(INPUT_N_IDX),
        self_sizes.at(INPUT_C_IDX),
        out_size.at(0),
        out_size.at(1),
        out_size.at(2)};
  } else if (scale_d != 1.0 || scale_h != 1.0 || scale_w != 1.0) {
    out_shape = {
        self_sizes.at(INPUT_N_IDX),
        self_sizes.at(INPUT_C_IDX),
        static_cast<int64_t>(
            static_cast<double>(self_sizes.at(INPUT_C_IDX)) * scale_d),
        static_cast<int64_t>(
            static_cast<double>(self_sizes.at(INPUT_H_IDX)) * scale_h),
        static_cast<int64_t>(
            static_cast<double>(self_sizes.at(INPUT_W_IDX)) * scale_w)};
  }
  return out_shape;
}
// Forward Meta Function - Nearest2D
OutputMetaDataVector UpsampleNearest2DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(2);
  std::vector<int64_t> out_shape;
  upsample_2d_common_check(self, out_size, scale);
  check_null_input(out_size, scale);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = UpsampleNearest2DFwdOutputShapeSynapseLayout(stack);

  check_input_output_height_width(
      self.sizes()[2], meta.shape.at(2), self.sizes()[3], meta.shape.at(3));
  return {meta};
}
// Backward Meta Function - Nearest2D
OutputMetaDataVector UpsampleNearest2DBwdMeta(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  OutputMetaData meta;
  meta.dtype = grad_in.scalar_type();
  meta.shape = stack.at(2).isTensor() ? stack_tensor(stack, 2).sizes().vec()
                                      : stack.at(2).toIntVector();
  check_null_input(out_size, scale);
  upsample_2d_common_check(grad_in, out_size, scale);
  return {meta};
}
// Forward Meta Function - NearestExact2D
OutputMetaDataVector UpsampleNearestExact2DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scales_h = stack.at(2).toOptional<double>();
  auto scales_w = stack.at(3).toOptional<double>();
  upsample_exact_2d_check(self, out_size);
  check_null_inputs_2d(out_size, scales_h, scales_w);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = UpsampleNearestExact2DFwdOutputShapeSynapseLayout(stack);

  check_input_output_height_width(
      self.sizes()[2], meta.shape.at(2), self.sizes()[3], meta.shape.at(3));
  return {meta};
}
// Backward Meta Function - NearestExact2D
OutputMetaDataVector UpsampleNearestExact2DBwdMeta(const at::Stack& stack) {
  auto grad_out = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto in_size = stack.at(2);
  auto scales_h = stack.at(3).toOptional<double>();
  auto scales_w = stack.at(4).toOptional<double>();
  OutputMetaData meta;
  meta.dtype = grad_out.scalar_type();
  meta.shape = in_size.toIntVector();
  check_null_inputs_2d(out_size, scales_h, scales_w);
  upsample_exact_2d_check(grad_out, out_size);
  return {meta};
}
// Forward Meta Function - NearestExact3D
OutputMetaDataVector UpsampleNearestExact3DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scales_d = stack.at(2).toOptional<double>();
  auto scales_h = stack.at(3).toOptional<double>();
  auto scales_w = stack.at(4).toOptional<double>();
  upsample_exact_3d_check(self, out_size);
  check_null_inputs_3d(out_size, scales_d, scales_h, scales_w);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = UpsampleNearestExact3DFwdOutputShapeSynapseLayout(stack);

  check_input_output_depth_height_width(
      self.sizes()[2],
      meta.shape.at(2),
      self.sizes()[3],
      meta.shape.at(3),
      self.sizes()[4],
      meta.shape.at(4));
  return {meta};
}

std::vector<int64_t> UpsampleBicubic2DFwdOutputShapeSynapseLayoutAA(
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale_h = stack.at(3).toOptional<double>().value_or(1.0);
  auto scale_w = stack.at(4).toOptional<double>().value_or(1.0);
  std::vector<int64_t> out_shape;
  if (!out_size.isNone()) {
    out_shape = {
        self.sizes()[INPUT_N_IDX],
        self.sizes()[INPUT_C_IDX],
        out_size.toIntVector().at(0),
        out_size.toIntVector().at(1)};
  } else if (scale_w != 1.0 || scale_h != 1.0) {
    out_shape = {
        self.sizes()[INPUT_N_IDX],
        self.sizes()[INPUT_C_IDX],
        static_cast<int64_t>(
            static_cast<double>(self.sizes()[INPUT_H_IDX]) * scale_h),
        static_cast<int64_t>(
            static_cast<double>(self.sizes()[INPUT_W_IDX]) * scale_w)};
  }
  return out_shape;
}

// Forward Meta Function - Bicubic2D AA
OutputMetaDataVector UpsampleBicubic2DFwdMetaAA(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale_h = stack.at(3).toOptional<double>();
  auto scale_w = stack.at(4).toOptional<double>();
  upsample_exact_2d_check(self, out_size);
  check_null_inputs_2d(out_size, scale_h, scale_w);

  OutputMetaData meta;
  meta.shape = UpsampleBicubic2DFwdOutputShapeSynapseLayoutAA(stack);
  meta.dtype = self.scalar_type();

  check_input_output_height_width(
      self.sizes()[2], meta.shape.at(2), self.sizes()[3], meta.shape.at(3));

  return {meta};
}
// Backward Meta Function - Bicubic2D AA
OutputMetaDataVector UpsampleBicubic2DBwdMetaAA(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale_h = stack.at(4).toOptional<double>();
  auto scale_w = stack.at(5).toOptional<double>();
  upsample_exact_2d_check(grad_in, out_size);
  check_null_inputs_2d(out_size, scale_h, scale_w);

  OutputMetaData meta;
  meta.shape = stack.at(2).toIntVector();
  meta.dtype = grad_in.scalar_type();
  return {meta};
}

// Forward Meta Function - Nearest3D
OutputMetaDataVector UpsampleNearest3DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(2);
  upsample_3d_common_check(self, out_size, scale);
  check_null_input(out_size, scale);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  if (!out_size.isNone()) {
    meta.shape = {
        self.sizes()[0],
        self.sizes()[1],
        out_size.toIntVector().at(0),
        out_size.toIntVector().at(1),
        out_size.toIntVector().at(2)};
  } else if (!scale.isNone()) {
    double scale_d = scale.toDoubleVector().at(0);
    double scale_h = scale.toDoubleVector().at(1);
    double scale_w = scale.toDoubleVector().at(2);
    meta.shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(static_cast<double>(self.sizes()[2]) * scale_d),
        static_cast<int64_t>(static_cast<double>(self.sizes()[3]) * scale_h),
        static_cast<int64_t>(static_cast<double>(self.sizes()[4]) * scale_w)};
  }
  check_input_output_depth_height_width(
      self.sizes()[2],
      meta.shape.at(2),
      self.sizes()[3],
      meta.shape.at(3),
      self.sizes()[4],
      meta.shape.at(4));
  return {meta};
}
// Backward Output Shape - Nearest3D
OutputMetaDataVector UpsampleNearest3DBwdMeta(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  check_null_input(out_size, scale);
  upsample_3d_common_check(grad_in, out_size, scale);

  OutputMetaData meta;
  meta.shape = stack.at(2).toIntVector();
  meta.dtype = grad_in.scalar_type();
  return {meta};
}
// Backward Output Shape - NearestExact3D
OutputMetaDataVector UpsampleNearestExact3DBwdMeta(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale_d = stack.at(3).toOptional<double>();
  auto scale_h = stack.at(4).toOptional<double>();
  auto scale_w = stack.at(5).toOptional<double>();
  check_null_inputs_3d(out_size, scale_d, scale_h, scale_w);
  upsample_exact_3d_check(grad_in, out_size);

  OutputMetaData meta;
  meta.shape = stack.at(2).toIntVector();
  meta.dtype = grad_in.scalar_type();
  return {meta};
}

SharedMetaDataVector UpsampleLinear1DFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return UpsampleCommmonSharedLayer(stack, stack.at(2).toBool(), 3, true);
}

SharedMetaDataVector UpsampleLinear1DBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return UpsampleCommmonSharedLayer(stack, stack.at(3).toBool(), 4, false);
}

SharedMetaDataVector UpsampleNearest1D3DFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return UpsampleCommmonSharedLayer(stack, false, 2, true);
}

SharedMetaDataVector UpsampleNearest1D3DBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return UpsampleCommmonSharedLayer(stack, false, 3, false);
}

SharedMetaDataVector UpsampleNearest2DFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return UpsampleCommmonSharedLayer(stack, true, 2, true);
}

SharedMetaDataVector UpsampleNearest2DBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return UpsampleCommmonSharedLayer(stack, true, 3, false);
}

FillParamsT FillBicubicFwdParamsAA(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(2).toBool();
  // scales
  auto scales = stack.at(3);
  double scale_h = stack.at(3).toOptional<double>().value_or(1.0);
  double scale_w = stack.at(4).toOptional<double>().value_or(1.0);
  double scale_d = 1.0;
  bool antialias = true;
  return FillResizeParams(
      self.dim(),
      bicubic,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

FillParamsT FillBicubicBwdParamsAA(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(3).toBool();
  // scales
  auto scales = stack.at(4);
  double scale_h = stack.at(4).toOptional<double>().value_or(1.0);
  double scale_w = stack.at(5).toOptional<double>().value_or(1.0);
  double scale_d = 1.0;
  bool antialias = true;
  return FillResizeParams(
      grad_in.dim(),
      bicubic,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

FillParamsT FillBilinearFwdParams(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(2).toBool();
  // scales
  auto scales = stack.at(3);
  double scale_w = 1.0;
  double scale_h = 1.0;
  double scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(3).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(4).toDouble();
  }
  return FillResizeParams(
      self.dim(),
      linear,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      false /*antialias*/);
}

std::tuple<double, double, double> ExtractScales(
    const at::IValue& scales,
    const at::Stack& stack,
    size_t scale_h_idx,
    size_t scale_w_idx) {
  double scale_w = 1.0;
  double scale_h = 1.0;
  double scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(scale_h_idx).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(scale_w_idx).toDouble();
  }
  return {scale_w, scale_h, scale_d};
}

FillParamsT FillBilinearParamsAAHelper(
    const at::Tensor& input_tensor,
    const at::Stack& stack,
    const at::IValue& out_size,
    const at::IValue& scales,
    bool align_corners,
    size_t scale_h_idx,
    size_t scale_w_idx) {
  auto [scale_w, scale_h, scale_d] =
      ExtractScales(scales, stack, scale_h_idx, scale_w_idx);
  return FillResizeParams(
      input_tensor.dim(),
      linear,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      true /*antialias*/);
}

FillParamsT FillBilinearFwdParamsAA(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(2).toBool();
  auto scales = stack.at(3);
  return FillBilinearParamsAAHelper(
      self, stack, out_size, scales, align_corners, 3, 4);
}

FillParamsT FillBilinearBwdParams(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(3).toBool();
  // scales
  auto scales = stack.at(4);
  double scale_w = 1.0;
  double scale_h = 1.0;
  double scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(4).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(5).toDouble();
  }
  return FillResizeParams(
      grad_in.dim(),
      linear,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      false /*antialias*/);
}

FillParamsT FillBilinearBwdParamsAA(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(3).toBool();
  auto scales = stack.at(4);
  return FillBilinearParamsAAHelper(
      grad_in, stack, out_size, scales, align_corners, 4, 5);
}

FillParamsT FillNearestFwdParams(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  // scales
  auto scales = stack.at(2);
  double scale_w = 1.0;
  double scale_h = 1.0;
  double scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(2).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(3).toDouble();
  }
  return FillResizeParams(
      self.dim(),
      nearest,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      false /*align_corners*/,
      false /*antialias*/);
}

FillParamsT FillNearestExact2DFwdParams(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  // scales
  auto scales_h = stack.at(2);
  auto scales_w = stack.at(3);
  double scale_w = scales_w.toOptional<double>().value_or(1.0);
  double scale_h = scales_h.toOptional<double>().value_or(1.0);
  double scale_d = 1.0;
  c10::IValue scales = scales_h;
  bool align_corners = false;
  bool antialias = false;
  return FillResizeParams(
      self.dim(),
      nearest_exact,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

FillParamsT FillNearestExact2DBwdParams(const at::Stack& stack) {
  auto grad_out = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scales_h = stack.at(3);
  auto scales_w = stack.at(4);
  bool align_corners = false;
  bool antialias = false;
  double scale_d = 1.0;
  double scale_w = scales_w.toOptional<double>().value_or(1.0);
  double scale_h = scales_h.toOptional<double>().value_or(1.0);
  return FillResizeParams(
      grad_out.dim(),
      nearest_exact,
      out_size,
      scales_h,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

FillParamsT FillNearestExact3DFwdParams(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  double scale_d = stack.at(2).toOptional<double>().value_or(1.0);
  double scale_h = stack.at(3).toOptional<double>().value_or(1.0);
  double scale_w = stack.at(4).toOptional<double>().value_or(1.0);
  c10::IValue scales = stack.at(2);
  bool align_corners = false;
  bool antialias = false;
  return FillResizeParams(
      self.dim(),
      nearest_exact,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

FillParamsT FillNearestBwdParams(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  // scales
  auto scales = stack.at(3);
  double scale_w = 1.0;
  double scale_h = 1.0;
  double scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(3).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(4).toDouble();
  }
  return FillResizeParams(
      grad_in.dim(),
      nearest,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      false /*align_corners*/,
      false /*antialias*/);
}

FillParamsT FillNearestExact3DBwdParams(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  bool align_corners = false;
  bool antialias = false;
  double scale_d = stack.at(3).toOptional<double>().value_or(1.0);
  double scale_h = stack.at(4).toOptional<double>().value_or(1.0);
  double scale_w = stack.at(5).toOptional<double>().value_or(1.0);

  return FillResizeParams(
      grad_in.dim(),
      nearest_exact,
      out_size,
      stack.at(3),
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

// AddNode FWD 1D Linear function
void UpsampleLinear1DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleLinear1DFwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  bool align_corners = stack.at(2).toBool();
  auto scales = stack.at(3);
  double scale_w = 1.0;
  if (!scales.isNone()) {
    scale_w =
        scales.isScalar() ? scales.toDouble() : scales.toDoubleVector().at(0);
  }
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      linear, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      {1.0 /*scale_d*/, 1.0 /*scale_h*/, scale_w},
      meta,
      self_tensor);
}
// AddNode BWD 1D Linear function
void UpsampleLinear1DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleLinear1DBwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  bool align_corners = stack.at(3).toBool();
  auto scales = stack.at(4);
  double scale_w = 1.0;
  if (!scales.isNone()) {
    scale_w =
        scales.isScalar() ? scales.toDouble() : scales.toDoubleVector().at(0);
  }
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      linear, /*upsample_mode*/
      false, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      {1.0 /*scale_d*/, 1.0 /*scale_h*/, scale_w},
      meta,
      self_tensor);
}
// AddNode FWD 1D Nearest function
void UpsampleNearest1DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleNearest1DFwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  bool align_corners = false;
  auto scales = stack.at(2);
  double scale_w = 1.0;
  if (!scales.isNone()) {
    scale_w =
        scales.isScalar() ? scales.toDouble() : scales.toDoubleVector().at(0);
  }
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      {1.0 /*scale_d*/, 1.0 /*scale_h*/, scale_w},
      meta,
      self_tensor);
}
// AddNode BWD 1D Nearest function
void UpsampleNearest1DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleNearest1DBwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  bool align_corners = false;
  auto scales = stack.at(3);
  double scale_w = 1.0;
  if (!scales.isNone()) {
    scale_w =
        scales.isScalar() ? scales.toDouble() : scales.toDoubleVector().at(0);
  }
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      false, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      {1.0 /*scale_d*/, 1.0 /*scale_h*/, scale_w},
      meta,
      self_tensor);
}
// AddNode FWD 1D Nearest Exact function
void UpsampleNearestExact1DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleNearest1DFwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  bool align_corners = false;
  auto scales = stack.at(2);
  double scale_w = 1.0;
  if (!scales.isNone()) {
    scale_w =
        scales.isScalar() ? scales.toDouble() : scales.toDoubleVector().at(0);
  }
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      nearest_exact, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      {1.0 /*scale_d*/, 1.0 /*scale_h*/, scale_w},
      meta,
      self_tensor);
}
// AddNode BWD 1D Nearest Exact function
void UpsampleNearestExact1DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleNearest1DBwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  bool align_corners = false;
  auto scales = stack.at(3);
  double scale_w = 1.0;
  if (!scales.isNone()) {
    scale_w =
        scales.isScalar() ? scales.toDouble() : scales.toDoubleVector().at(0);
  }
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      nearest_exact, /*upsample_mode*/
      false, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      {1.0 /*scale_d*/, 1.0 /*scale_h*/, scale_w},
      meta,
      self_tensor);
}
//  AddNode 2D Nearest function
void UpSampleNearest2DOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  auto self = stack_tensor(stack, 0);
  std::vector<synTensor> input{syn_in(0)};
  std::optional<synapse_helpers::tensor> cast_storage;
  std::optional<int> final_index = 0;
  CreateShapeTensorInput(graph, meta.dtype, meta.shape, input, SHAPE_TENSOR);
  auto intermediateDtype = meta.dtype;
  if (meta.dtype == c10::ScalarType::Byte) {
    // u8 to f32
    intermediateDtype = c10::ScalarType::Float;
    cast_storage = BuildCast(
        this,
        graph,
        input[0],
        self.sizes().vec(),
        meta.dtype,
        intermediateDtype);
    input[0] = cast_storage->get();
    final_index = std::nullopt;
  }

  const auto& params = FillParams(stack);

  auto resize = Resize(
      this, graph, input, meta.shape, intermediateDtype, params, final_index);
  if (meta.dtype == c10::ScalarType::Byte) {
    // f32 to u8
    resize[0] = BuildCast(
        this,
        graph,
        resize[0].get(),
        meta.shape,
        intermediateDtype,
        meta.dtype,
        0);
  }
  syn_out(0) = std::move(resize.at(0));
}

synapse_helpers::tensor UpsampleNearestExactFwdCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<synTensor> input,
    const FillParamsT& params) {
  auto meta = op->OutputMeta(stack)[0];
  auto self = stack_tensor(stack, 0);
  std::optional<synapse_helpers::tensor> cast_storage;
  std::optional<int> final_index = 0;
  op->CreateShapeTensorInput(
      graph, meta.dtype, meta.shape, input, SHAPE_TENSOR);
  auto intermediateDtype = meta.dtype;
  if (meta.dtype == c10::ScalarType::Byte) {
    // u8 to f32
    intermediateDtype = c10::ScalarType::Float;
    cast_storage = OpBackend::BuildCast(
        op, graph, input[0], self.sizes().vec(), meta.dtype, intermediateDtype);
    input[0] = cast_storage->get();
    final_index = std::nullopt;
  }

  auto resize = Resize(
      op, graph, input, meta.shape, intermediateDtype, params, final_index);
  if (meta.dtype != c10::ScalarType::Byte)
    return std::move(resize[0]);

  // f32 to u8
  return OpBackend::BuildCast(
      op, graph, resize[0].get(), meta.shape, intermediateDtype, meta.dtype, 0);
}
// AddNode FWD 2D Nearest Exact function
void UpsampleNearestExact2DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto params = FillParams(stack);
  syn_out(0) =
      UpsampleNearestExactFwdCommon(this, graph, stack, {syn_in(0)}, params);
}
// AddNode FWD 3D Nearest Exact function
void UpsampleNearestExact3DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto params = FillParams(stack);
  syn_out(0) =
      UpsampleNearestExactFwdCommon(this, graph, stack, {syn_in(0)}, params);
}

// AddNode BWD 2D Nearest Exact function
void UpsampleNearestExact2DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleNearestExact2DBwdMeta(stack)[0];
  std::optional<int> final_index = 0;

  const auto& params = FillParams(stack);

  auto resize = Resize(
      this, graph, {syn_in(0)}, meta.shape, meta.dtype, params, final_index);

  syn_out(0) = std::move(resize.at(0));
}
// AddNode FWD 3D Nearest function
void UpSampleNearest3DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleNearest3DFwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  // scales
  auto scales = stack.at(2);
  double scale_d = 1.0;
  double scale_w = 1.0;
  double scale_h = 1.0;
  if (!scales.isNone()) {
    scale_d = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(2).toDouble();
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(3).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(2)
                                 : stack.at(4).toDouble();
  }
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      out_size,
      false, /*align_corners*/
      scales,
      {scale_d, scale_h, scale_w},
      meta,
      self_tensor);
}
// AddNode BWD 3D Nearest function
void UpSampleNearest3DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // outshape
  auto meta = UpsampleNearest3DBwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  // scales
  auto scales = stack.at(3);
  double scale_d = 1.0;
  double scale_w = 1.0;
  double scale_h = 1.0;
  if (!scales.isNone()) {
    scale_d = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(3).toDouble();
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(4).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(2)
                                 : stack.at(5).toDouble();
  }
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      false, /*isForward*/
      {syn_in(0)},
      out_size,
      false, /*align_corners*/
      scales,
      {scale_d, scale_h, scale_w},
      meta,
      self_tensor);
}
// AddNode BWD 3D Nearest Exact function
void UpsampleNearestExact3DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // outshape
  auto meta = UpsampleNearestExact3DBwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  // scales
  auto scales_d = stack.at(3);
  double scale_d = scales_d.toOptional<double>().value_or(1.0);
  double scale_h = stack.at(4).toOptional<double>().value_or(1.0);
  double scale_w = stack.at(5).toOptional<double>().value_or(1.0);
  bool isForward = false;
  bool align_corners = false;
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      nearest_exact,
      isForward,
      {syn_in(0)},
      out_size,
      align_corners,
      scales_d,
      {scale_d, scale_h, scale_w},
      meta,
      self_tensor);
}

} // namespace habana
