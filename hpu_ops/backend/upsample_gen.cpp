/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "backend/synapse_helpers/layout_utils.h"
#include "generated/backend/_upsample_nearest_exact1d.h"
#include "generated/backend/_upsample_nearest_exact1d_backward.h"
#include "generated/backend/upsample_linear1d.h"
#include "generated/backend/upsample_trilinear3d.h"
#include "generated/backend/upsample_linear1d_backward.h"
#include "generated/backend/upsample_nearest1d.h"
#include "generated/backend/upsample_nearest1d_backward.h"
#include "generated/backend/upsample_nearest2d.h"
#include "generated/backend/upsample_nearest2d_backward.h"
#include "generated/backend/upsample_nearest3d.h"
#include "generated/backend/upsample_nearest3d_backward.h"

using namespace synapse_helpers::layouts;

#define CHECK_NULL_INPUT(out_size, scale)                     \
  TORCH_CHECK(                                                \
      !(out_size == c10::nullopt && scale == c10::nullopt) || \
          (out_size != c10::nullopt &&                        \
           (scale != c10::nullopt && !scale.isScalar())),     \
      "Upsample: Must specify exactly one of output_size and scale_factors");

#define CHECK_INPUT_OUTPUT_WIDTH(input_width, output_width)                               \
  TORCH_CHECK(                                                                            \
      input_width > 0 && output_width > 0,                                                \
      "Upsample1D:  Input and output sizes should be greater than 0, but got input (W: ", \
      input_width,                                                                        \
      ") and output (W: ",                                                                \
      output_width,                                                                       \
      ")");

#define CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(                                                  \
    input_height, output_height, input_width, output_width)                               \
  TORCH_CHECK(                                                                            \
      (input_width > 0 && output_width > 0) &&                                            \
          (input_height > 0 && output_height > 0),                                        \
      "Upsample2D:  Input and output sizes should be greater than 0, but got input (W: ", \
      input_width,                                                                        \
      ") and (H: ",                                                                       \
      input_height,                                                                       \
      ") output (W: ",                                                                    \
      output_width,                                                                       \
      ") for Upsample2D");

#define CHECK_INPUT_OUTPUT_DEPTH_HEIGHT_WIDTH(                                           \
    input_depth,                                                                         \
    output_depth,                                                                        \
    input_height,                                                                        \
    output_height,                                                                       \
    input_width,                                                                         \
    output_width)                                                                        \
  TORCH_CHECK(                                                                           \
      (input_depth > 0 && output_depth > 0) &&                                           \
          (input_width > 0 && output_width > 0) &&                                       \
          (input_height > 0 && output_height > 0),                                       \
      "Upsample3D: Input and output sizes should be greater than 0, but got input (W: ", \
      input_width,                                                                       \
      ") and (H: ",                                                                      \
      input_height,                                                                      \
      ") and (D: ",                                                                      \
      input_depth,                                                                       \
      ")                                                                                 \
      output (W: ",                                                                      \
      output_width,                                                                      \
      ") and (H: ",                                                                      \
      output_height,                                                                     \
      ")                                                                                 \
      and (D: ",                                                                         \
      output_depth,                                                                      \
      ") for Upsample3D");

namespace habana {
// Upsample1D Common checks
void upsample_1d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales) {
  TORCH_CHECK(
      input.dim() == 3,
      "Upsample1D expects input_size equals to 3, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    TORCH_CHECK(
        out_size.toIntVector().size() == 1,
        "Upsample1D expects out_size equals to 1, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    TORCH_CHECK(
        scales.toDoubleVector().size() == 1,
        "Upsample1D expects scales equals to 1, but got ",
        scales.toDoubleVector().size());
  }
}
// Upsample2D Common checks
void upsample_2d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales) {
  TORCH_CHECK(
      input.dim() == 4,
      "Upsample2D expects input_size equals to 4, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    TORCH_CHECK(
        out_size.toIntVector().size() == 2,
        "Upsample2D expects out_size equals to 2, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    TORCH_CHECK(
        scales.toDoubleVector().size() == 2,
        "Upsample2D expects scales equals to 2, but got ",
        scales.toDoubleVector().size());
  }
}

void upsample_3d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales) {
  TORCH_CHECK(
      input.dim() == 5,
      "Upsample3D expects input_size equals to 5, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    TORCH_CHECK(
        out_size.toIntVector().size() == 3,
        "Upsample3D expects out_size equals to 3, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    TORCH_CHECK(
        scales.toDoubleVector().size() == 3,
        "Upsample3D expects scales equals to 3, but got ",
        scales.toDoubleVector().size());
  }
}

// Forward Meta Function - Linear1D
OutputMetaDataVector UpsampleLinear1DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  upsample_1d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  if (!out_size.isNone()) {
    meta.shape = {
        self.sizes()[0], self.sizes()[1], out_size.toIntVector().at(0)};
  } else if (!scale.isNone() && !scale.isScalar()) {
    double scale_factor = scale.toDoubleVector().at(0);
    auto width = self.sizes()[2];
    meta.shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(width * scale_factor)};
  }
  CHECK_INPUT_OUTPUT_WIDTH(self.sizes()[2], meta.shape.at(2));
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
  CHECK_NULL_INPUT(out_size, scale);
  if (!out_size.isNone()) {
    meta.shape = {
        self.sizes()[0], self.sizes()[1], out_size.toIntVector().at(0)};
  } else if (!scale.isNone()) {
    double scale_factor = scale.toDoubleVector().at(0);
    auto width = self.sizes()[2];
    meta.shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(width * scale_factor)};
  }
  CHECK_INPUT_OUTPUT_WIDTH(self.sizes()[2], meta.shape.at(2));
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
        static_cast<int64_t>(self.sizes()[INPUT_H_IDX] * scale_h),
        static_cast<int64_t>(self.sizes()[INPUT_W_IDX] * scale_w)};
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
  CHECK_NULL_INPUT(out_size, scale);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = UpsampleBilinear2DFwdOutputShapeSynapseLayout(stack);

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
      self.sizes()[2], meta.shape.at(2), self.sizes()[3], meta.shape.at(3));
  return {meta};
}
// Backward Meta Function - Bilinear2D
OutputMetaDataVector UpsampleBilinear2DBwdMeta(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(4);
  CHECK_NULL_INPUT(out_size, scale);
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
        static_cast<int64_t>(self.sizes()[INPUT_H_IDX] * scale_h),
        static_cast<int64_t>(self.sizes()[INPUT_W_IDX] * scale_w)};
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
  CHECK_NULL_INPUT(out_size, scale)
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = UpsampleNearest2DFwdOutputShapeSynapseLayout(stack);

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
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
  CHECK_NULL_INPUT(out_size, scale);
  upsample_2d_common_check(grad_in, out_size, scale);
  return {meta};
}

std::vector<int64_t> UpsampleBicubic2DFwdOutputShapeSynapseLayout(
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
        static_cast<int64_t>(self.sizes()[INPUT_H_IDX] * scale_h),
        static_cast<int64_t>(self.sizes()[INPUT_W_IDX] * scale_w)};
  }
  return out_shape;
}
// Forward Meta Function - Bicubic2D
OutputMetaDataVector UpsampleBicubic2DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  std::vector<int64_t> out_shape;
  upsample_2d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);

  OutputMetaData meta;
  meta.shape = UpsampleBicubic2DFwdOutputShapeSynapseLayout(stack);
  meta.dtype = self.scalar_type();

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
      self.sizes()[2], meta.shape.at(2), self.sizes()[3], meta.shape.at(3));

  return {meta};
}
// Backward Meta Function - Bicubic2D
OutputMetaDataVector UpsampleBicubic2DBwdMeta(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(4);

  OutputMetaData meta;
  meta.shape = stack.at(2).toIntVector();
  meta.dtype = grad_in.scalar_type();
  CHECK_NULL_INPUT(out_size, scale);
  upsample_2d_common_check(grad_in, out_size, scale);
  return {meta};
}
OutputMetaDataVector UpsampleTrilinear3DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  upsample_3d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);
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
        static_cast<int64_t>(self.sizes()[2] * scale_d),
        static_cast<int64_t>(self.sizes()[3] * scale_h),
        static_cast<int64_t>(self.sizes()[4] * scale_w)};
  }
  CHECK_INPUT_OUTPUT_DEPTH_HEIGHT_WIDTH(
      self.sizes()[2],
      meta.shape.at(2),
      self.sizes()[3],
      meta.shape.at(3),
      self.sizes()[4],
      meta.shape.at(4));
  return {meta};
}
// Forward Meta Function - Nearest3D
OutputMetaDataVector UpsampleNearest3DFwdMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(2);
  upsample_3d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);
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
        static_cast<int64_t>(self.sizes()[2] * scale_d),
        static_cast<int64_t>(self.sizes()[3] * scale_h),
        static_cast<int64_t>(self.sizes()[4] * scale_w)};
  }
  CHECK_INPUT_OUTPUT_DEPTH_HEIGHT_WIDTH(
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
  CHECK_NULL_INPUT(out_size, scale);
  upsample_3d_common_check(grad_in, out_size, scale);

  OutputMetaData meta;
  meta.shape = stack.at(2).toIntVector();
  meta.dtype = grad_in.scalar_type();
  return {meta};
}

enum modes { nearest, nearest_exact, linear, bicubic };

// Custom FillParams function
std::shared_ptr<void> FillResizeParams(
    const int variant_type,
    size_t& size,
    enum modes upsample_mode,
    c10::IValue out_size,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    bool align_corner) {
  PARAMS_STUB(ns_ResizeKernel::Params);
  params->excludeOutside = false;
  switch (upsample_mode) {
    case nearest:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
      params->nearestMode = ResizeNearestMode_t::FLOOR;
      params->coordTransMode =
          ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
      break;
    case nearest_exact:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
      params->nearestMode = ResizeNearestMode_t::ROUND_DEFAULT;
      params->coordTransMode =
          ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
      break;
    case linear:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_LINEAR;
      params->nearestMode = ResizeNearestMode_t::FLOOR;
      params->coordTransMode = align_corner
          ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
          : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
      break;
    case bicubic:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_CUBIC;
      params->nearestMode = ResizeNearestMode_t::ROUND_DEFAULT;
      params->coordTransMode = align_corner
          ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
          : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
      params->cubicCoeffA =
          -0.75; // As mentioned in TPC guide, value of cubicCoeffA used for
                 // cubic interpolation is -0.75.
      break;
  }
  if (!out_size.isNone()) {
    params->useScales = false;
    if (variant_type == 3) { // 1D variant
      params->size1 = out_size.toIntVector().at(0);
    } else if (variant_type == 4) { // 2D variant
      params->size1 = out_size.toIntVector().at(1);
      params->size2 = out_size.toIntVector().at(0);
    } else if (variant_type == 5) { // 3D variant
      params->size1 = out_size.toIntVector().at(2);
      params->size2 = out_size.toIntVector().at(1);
      params->size3 = out_size.toIntVector().at(0);
    }
    if (align_corner) {
      return params;
    }
  }
  if (!scales.isNone()) {
    params->useScales = true;
    params->scaleDim1 = scale_w;
    params->scaleDim2 = scale_h;
    params->scaleDim3 = scale_d;
  }
  return params;
}

std::shared_ptr<void> FillBicubicFwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(2).toBool();
  // scales
  auto scales = stack.at(3);
  double scale_w = 1.0, scale_h = 1.0, scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(3).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(4).toDouble();
  }
  return FillResizeParams(
      self.dim(),
      size,
      bicubic,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners);
}

std::shared_ptr<void> FillBicubicBwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(3).toBool();
  // scales
  auto scales = stack.at(4);
  double scale_w = 1.0, scale_h = 1.0, scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(4).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(5).toDouble();
  }
  return FillResizeParams(
      grad_in.dim(),
      size,
      bicubic,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners);
}

std::shared_ptr<void> FillBilinearFwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(2).toBool();
  // scales
  auto scales = stack.at(3);
  double scale_w = 1.0, scale_h = 1.0, scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(3).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(4).toDouble();
  }
  return FillResizeParams(
      self.dim(),
      size,
      linear,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners);
}

std::shared_ptr<void> FillBilinearBwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(3).toBool();
  // scales
  auto scales = stack.at(4);
  double scale_w = 1.0, scale_h = 1.0, scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(4).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(5).toDouble();
  }
  return FillResizeParams(
      grad_in.dim(),
      size,
      linear,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners);
}

std::shared_ptr<void> FillNearestFwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  // scales
  auto scales = stack.at(2);
  double scale_w = 1.0, scale_h = 1.0, scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(2).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(3).toDouble();
  }
  return FillResizeParams(
      self.dim(),
      size,
      nearest,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      false /*align_corners*/);
}

std::shared_ptr<void> FillNearestBwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  // scales
  auto scales = stack.at(3);
  double scale_w = 1.0, scale_h = 1.0, scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(3).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(4).toDouble();
  }
  return FillResizeParams(
      grad_in.dim(),
      size,
      nearest,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      false /*align_corners*/);
}

// Resize TPC kernel
static std::vector<synapse_helpers::tensor> Resize(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    const at::ScalarType& dtype,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_index = c10::nullopt) {
  auto guid = op->GetGuid();
  update_guid_dtype(guid, dtype);

  return OpBackend::BuildNode(
      op,
      graph,
      {guid,
       std::move(input),
       {{outshape, dtype, final_index}},
       params.get(),
       size});
}
// Slice the result when both scale and size provided - outplace fwd varaint
static std::vector<synapse_helpers::tensor> Slice(
    OpBackend* op,
    synapse_helpers::graph& graph,
    int64_t input_size,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    const at::ScalarType& dtype,
    c10::optional<int> final_index = c10::nullopt) {
  if (input_size == 3) {
    // 3D inputs are reshaped to 4D inputs
    input_size = 4;
  }
  synSliceParamsV2 slice_params{};
  for (int64_t i = input_size - 1; i >= 0; --i) {
    slice_params.axes[i] = i;
    slice_params.starts[i] = 0;
    slice_params.ends[i] = outshape[(input_size - i - 1)];
    slice_params.steps[i] = 1;
  }
  return OpBackend::BuildNode(
      op,
      graph,
      {"slice",
       std::move(input),
       {{outshape, dtype, final_index}},
       &slice_params,
       sizeof(slice_params)});
}

// Upsample Common function - New Layout
std::vector<synapse_helpers::tensor> UpsampleCommonFuncSynapseLayout(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes upsample_mode,
    bool isForward,
    std::vector<synTensor> input,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    const OutputMetaData& meta,
    const at::Tensor self_tensor) {
  auto variant_type = self_tensor.dim();
  auto shape_in = self_tensor.sizes().vec();
  std::vector<int64_t> out_shape_temp(shape_in.begin(), shape_in.end());

  std::unique_ptr<synapse_helpers::tensor> cast;
  auto intermediateDtype = meta.dtype;
  if (meta.dtype == c10::ScalarType::Byte) {
    // u8 to f32
    cast = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
        op,
        graph,
        input[0],
        shape_in,
        c10::ScalarType::Byte,
        c10::ScalarType::Float));
    input = {cast->get()};
    intermediateDtype = c10::ScalarType::Float;
  }
  // Resize
  // modify input width value with output width value
  // when both size and scale is provided with align_corners=false
  bool modifyInputWithOutputWidth =
      isForward && !align_corners && (!out_size.isNone() && !scales.isNone());
  if (modifyInputWithOutputWidth) {
    if (variant_type == 3) { // 1D
      out_shape_temp.at(2) = static_cast<int64_t>(shape_in[2] * scale_w);
    } else if (variant_type == 5) { // 3D
      out_shape_temp.at(2) = static_cast<int64_t>(shape_in[2] * scale_d);
      out_shape_temp.at(3) = static_cast<int64_t>(shape_in[3] * scale_h);
      out_shape_temp.at(4) = static_cast<int64_t>(shape_in[4] * scale_w);
    }
  } else {
    if (variant_type == 3) { // 1D
      out_shape_temp.at(2) = meta.shape.at(2);
    } else if (variant_type == 5) { // 3D
      out_shape_temp.at(2) = meta.shape.at(2);
      out_shape_temp.at(3) = meta.shape.at(3);
      out_shape_temp.at(4) = meta.shape.at(4);
    }
  }

  size_t size = 0;
  const auto& params = FillResizeParams(
      variant_type,
      size,
      upsample_mode,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners);
  auto final_index_for_resize = modifyInputWithOutputWidth ||
          meta.dtype == c10::ScalarType::Byte
      ? c10::optional<int>()
      : c10::optional<int>(0);
  auto resize = Resize(
      op,
      graph,
      input,
      out_shape_temp,
      intermediateDtype,
      params,
      size,
      final_index_for_resize);
  // Slice
  // For Fwd ops, when both size and scale is provided with align_corners=false
  if (modifyInputWithOutputWidth) {
    std::vector<int64_t> slice_shape(meta.shape.begin(), meta.shape.end());
    if (variant_type == 3) { // 1D
      // NCW
      slice_shape = {shape_in[0], shape_in[1], meta.shape.at(2)};
    }
    auto final_index_for_slice =
        (meta.dtype == c10::ScalarType::Byte)
        ? c10::optional<int>()
        : c10::optional<int>(0);

    resize = Slice(
        op,
        graph,
        variant_type,
        {resize[0].get()},
        slice_shape,
        intermediateDtype,
        final_index_for_slice);
  };
  if (meta.dtype == c10::ScalarType::Byte) {
    // f32 to u8
    std::vector<synapse_helpers::tensor> result;
    result.emplace_back(OpBackend::BuildCast(
        op,
        graph,
        resize[0].get(),
        meta.shape,
        intermediateDtype,
        meta.dtype,
        0));
    return result;
  }
  return resize;
}

// Upsample Common function
std::vector<synapse_helpers::tensor> UpsampleCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes upsample_mode,
    bool isForward,
    std::vector<synTensor> input,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    const OutputMetaData& meta,
    const at::Tensor self_tensor) {
  PT_LAZY_DEBUG(__FUNCTION__);
  std::vector<synapse_helpers::tensor> output;
  op->CreateShapeTensorInput(
      graph, meta.dtype, meta.shape, input, SHAPE_TENSOR);
  return UpsampleCommonFuncSynapseLayout(
      op,
      graph,
      upsample_mode,
      isForward,
      input,
      out_size,
      align_corners,
      scales,
      scale_w,
      scale_h,
      scale_d,
      meta,
      self_tensor);
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
  auto result = UpsampleCommonFunc(
      this,
      graph,
      linear, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      meta,
      self_tensor);

  syn_out(0) = std::move(result.at(0));
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
  auto result = UpsampleCommonFunc(
      this,
      graph,
      linear, /*upsample_mode*/
      false, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      meta,
      self_tensor);
  syn_out(0) = std::move(result.at(0));
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
  auto result = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      meta,
      self_tensor);
  syn_out(0) = std::move(result.at(0));
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
  auto result = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      false, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      meta,
      self_tensor);
  syn_out(0) = std::move(result.at(0));
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
  auto result = UpsampleCommonFunc(
      this,
      graph,
      nearest_exact, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      meta,
      self_tensor);
  syn_out(0) = std::move(result.at(0));
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
  auto result = UpsampleCommonFunc(
      this,
      graph,
      nearest_exact, /*upsample_mode*/
      false, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      meta,
      self_tensor);
  syn_out(0) = std::move(result.at(0));
}
//  AddNode 2D Nearest function
void UpSampleNearest2DOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  auto self = stack_tensor(stack, 0);
  std::vector<synTensor> input{syn_in(0)};
  std::unique_ptr<synapse_helpers::tensor> cast;
  c10::optional<int> final_index = 0;
  CreateShapeTensorInput(graph, meta.dtype, meta.shape, input, SHAPE_TENSOR);
  auto intermediateDtype = meta.dtype;
  if (meta.dtype == c10::ScalarType::Byte) {
    // u8 to f32
    intermediateDtype = c10::ScalarType::Float;
    cast = std::make_unique<synapse_helpers::tensor>(BuildCast(
        this,
        graph,
        input[0],
        self.sizes().vec(),
        meta.dtype,
        intermediateDtype));
    input = {cast->get()};
    final_index = c10::nullopt;
  }

  size_t size = 0;
  const auto& params = FillParams(stack, size);

  auto resize = Resize(
      this,
      graph,
      input,
      meta.shape,
      intermediateDtype,
      params,
      size,
      final_index);
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
void UpSampleTrilinear3DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleTrilinear3DFwdMeta(stack)[0];
  auto self_tensor = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(2).toBool();

  auto scales = stack.at(3);
  double scale_d = 1.0, scale_w = 1.0, scale_h = 1.0;
  if (!scales.isNone()) {
    scale_d = stack.at(3).toOptional<double>().value_or(1.0f);
    scale_h = stack.at(4).toOptional<double>().value_or(1.0f);
    scale_w = stack.at(5).toOptional<double>().value_or(1.0f);
  }
  std::vector<synTensor> input = {syn_in(0)};
  auto result = UpsampleCommonFunc(
      this,
      graph,
      linear, /*upsample_mode*/
      true, /*isForward*/
      input,
      out_size,
      align_corners,
      scales,
      scale_w,
      scale_h,
      scale_d,
      meta,
      self_tensor);
  syn_out(0) = std::move(result.at(0));
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
  double scale_d = 1.0, scale_w = 1.0, scale_h = 1.0;
  if (!scales.isNone()) {
    scale_d = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(2).toDouble();
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(3).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(2)
                                 : stack.at(4).toDouble();
  }
  std::vector<synTensor> input = {syn_in(0)};
  auto result = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      true, /*isForward*/
      input,
      out_size,
      false, /*align_corners*/
      scales,
      scale_w,
      scale_h,
      scale_d,
      meta,
      self_tensor);
  syn_out(0) = std::move(result.at(0));
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
  double scale_d = 1.0, scale_w = 1.0, scale_h = 1.0;
  if (!scales.isNone()) {
    scale_d = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(3).toDouble();
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(4).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(2)
                                 : stack.at(5).toDouble();
  }
  std::vector<synTensor> input = {syn_in(0)};
  auto result = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      false, /*isForward*/
      input,
      out_size,
      false, /*align_corners*/
      scales,
      scale_w,
      scale_h,
      scale_d,
      meta,
      self_tensor);
  syn_out(0) = std::move(result.at(0));
}
} // namespace habana
