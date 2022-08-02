/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/upsample_linear1d.h"
#include "generated/upsample_linear1d_backward.h"
#include "generated/upsample_nearest1d.h"
#include "generated/upsample_nearest1d_backward.h"
#include "generated/upsample_nearest3d.h"
#include "generated/upsample_nearest3d_backward.h"
#include "synapse_helpers/layout_utils.h"

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

// Forward Output Shape - Linear1D
sizes_vec UpsampleLinear1DFwdOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  std::vector<int64_t> out_shape;
  upsample_1d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);
  if (!out_size.isNone()) {
    out_shape = {
        self.sizes()[0], self.sizes()[1], out_size.toIntVector().at(0)};
  } else if (!scale.isNone() && !scale.isScalar()) {
    double scale_factor = scale.toDoubleVector().at(0);
    auto width = self.sizes()[2];
    out_shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(width * scale_factor)};
  }
  CHECK_INPUT_OUTPUT_WIDTH(self.sizes()[2], out_shape.at(2));
  return {out_shape};
}
// Backward Output Shape - Linear1D
sizes_vec UpsampleLinear1DBwdOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(4);
  upsample_1d_common_check(self, out_size, scale);
  return {stack.at(2).toIntVector()};
}
// Forward Output Shape - Nearest1D
sizes_vec UpsampleNearest1DFwdOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(2);
  std::vector<int64_t> out_shape;
  upsample_1d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);
  if (!out_size.isNone()) {
    out_shape = {
        self.sizes()[0], self.sizes()[1], out_size.toIntVector().at(0)};
  } else if (!scale.isNone()) {
    double scale_factor = scale.toDoubleVector().at(0);
    auto width = self.sizes()[2];
    out_shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(width * scale_factor)};
  }
  CHECK_INPUT_OUTPUT_WIDTH(self.sizes()[2], out_shape.at(2));
  return {out_shape};
}
// Backward Output Shape - Nearest1D
sizes_vec UpsampleNearest1DBwdOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  upsample_1d_common_check(self, out_size, scale);
  return {stack.at(2).toIntVector()};
}
// Forward Output Shape - Bilinear2D
std::vector<int64_t> UpsampleBilinear2DFwdOutputShapeSynapseLayout(
    const at::Stack& stack) {
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape for Synapse layout handling mode");
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
// Forward Output Shape - Bilinear2D
sizes_vec UpsampleBilinear2DFwdOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  std::vector<int64_t> out_shape;
  upsample_2d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == true) {
    out_shape = UpsampleBilinear2DFwdOutputShapeSynapseLayout(stack);
  } else {
    if (!habana_lazy::isDeviceInLoweringMode()) {
      if (!out_size.isNone()) {
        out_shape = {
            self.sizes()[0],
            self.sizes()[1],
            out_size.toIntVector().at(0),
            out_size.toIntVector().at(1)};
      } else if (!scale.isNone()) {
        double scale_w = scale.toDoubleVector().at(1);
        double scale_h = scale.toDoubleVector().at(0);
        out_shape = {
            self.sizes()[0],
            self.sizes()[1],
            static_cast<int64_t>(self.sizes()[2] * scale_h),
            static_cast<int64_t>(self.sizes()[3] * scale_w)};
      }
    } else {
      if (!out_size.isNone()) {
        out_shape = {
            self.sizes()[0],
            out_size.toIntVector().at(0),
            out_size.toIntVector().at(1),
            self.sizes()[3]};
      } else if (!scale.isNone()) {
        double scale_w = scale.toDoubleVector().at(1);
        double scale_h = scale.toDoubleVector().at(0);
        out_shape = {
            self.sizes()[0],
            static_cast<int64_t>(self.sizes()[1] * scale_h),
            static_cast<int64_t>(self.sizes()[2] * scale_w),
            self.sizes()[3]};
      }
    }
  }

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
      self.sizes()[2], out_shape.at(2), self.sizes()[3], out_shape.at(3));
  return {out_shape};
}
// Backward Output Shape - Bilinear2D
sizes_vec UpsampleBilinear2DBwdOutputShape(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(4);
  std::vector<int64_t> outshape = stack.at(2).toIntVector();
  CHECK_NULL_INPUT(out_size, scale);
  upsample_2d_common_check(grad_in, out_size, scale);
  if (!habana_lazy::isDeviceInLoweringMode() ||
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == true) {
    return {outshape};
  } else { // always in NHWC - backend
    return {{outshape[0], outshape[2], outshape[3], outshape[1]}};
  }
}
std::vector<int64_t> UpsampleNearest2DFwdOutputShapeSynapseLayout(
    const at::Stack& stack) {
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape for Synapse layout handling mode");
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
// Forward Output Shape - Nearest2D
sizes_vec UpsampleNearest2DFwdOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(2);
  std::vector<int64_t> out_shape;
  upsample_2d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale)

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == true) {
    out_shape = UpsampleNearest2DFwdOutputShapeSynapseLayout(stack);
  } else {
    if (!habana_lazy::isDeviceInLoweringMode()) {
      // NCHW
      if (!out_size.isNone()) {
        out_shape = {
            self.sizes()[0],
            self.sizes()[1],
            out_size.toIntVector().at(0),
            out_size.toIntVector().at(1)};
      } else if (!scale.isNone()) {
        double scale_w = scale.toDoubleVector().at(1);
        double scale_h = scale.toDoubleVector().at(0);
        out_shape = {
            self.sizes()[0],
            self.sizes()[1],
            static_cast<int64_t>(self.sizes()[2] * scale_h),
            static_cast<int64_t>(self.sizes()[3] * scale_w)};
      }
    } else {
      if (!out_size.isNone()) {
        out_shape = {
            self.sizes()[0],
            out_size.toIntVector().at(0),
            out_size.toIntVector().at(1),
            self.sizes()[3]};
      } else if (!scale.isNone()) {
        double scale_w = scale.toDoubleVector().at(1);
        double scale_h = scale.toDoubleVector().at(0);
        out_shape = {
            self.sizes()[0],
            static_cast<int64_t>(self.sizes()[1] * scale_h),
            static_cast<int64_t>(self.sizes()[2] * scale_w),
            self.sizes()[3]};
      }
    }
  }

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
      self.sizes()[2], out_shape.at(2), self.sizes()[3], out_shape.at(3));
  return {out_shape};
}
// Backward Output Shape - Nearest2D
sizes_vec UpsampleNearest2DBwdOutputShape(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  std::vector<int64_t> outshape = stack.at(2).toIntVector();
  CHECK_NULL_INPUT(out_size, scale);
  upsample_2d_common_check(grad_in, out_size, scale);
  if (!habana_lazy::isDeviceInLoweringMode() ||
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == true) {
    return {outshape};
  } else {
    return {{outshape[0], outshape[2], outshape[3], outshape[1]}};
  }
}

std::vector<int64_t> UpsampleBicubic2DFwdOutputShapeSynapseLayout(
    const at::Stack& stack) {
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape for Synapse layout handling mode");
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
// Forward Output Shape - Bicubic2D
sizes_vec UpsampleBicubic2DFwdOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  std::vector<int64_t> out_shape;
  upsample_2d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == true) {
    out_shape = UpsampleBicubic2DFwdOutputShapeSynapseLayout(stack);
  } else {
    if (!habana_lazy::isDeviceInLoweringMode()) {
      if (!out_size.isNone()) {
        out_shape = {
            self.sizes()[0],
            self.sizes()[1],
            out_size.toIntVector().at(0),
            out_size.toIntVector().at(1)};
      } else if (!scale.isNone()) {
        double scale_w = scale.toDoubleVector().at(1);
        double scale_h = scale.toDoubleVector().at(0);
        out_shape = {
            self.sizes()[0],
            self.sizes()[1],
            static_cast<int64_t>(self.sizes()[2] * scale_h),
            static_cast<int64_t>(self.sizes()[3] * scale_w)};
      }
    } else {
      if (!out_size.isNone()) {
        out_shape = {
            self.sizes()[0],
            out_size.toIntVector().at(0),
            out_size.toIntVector().at(1),
            self.sizes()[3]};
      } else if (!scale.isNone()) {
        double scale_w = scale.toDoubleVector().at(1);
        double scale_h = scale.toDoubleVector().at(0);
        out_shape = {
            self.sizes()[0],
            static_cast<int64_t>(self.sizes()[1] * scale_h),
            static_cast<int64_t>(self.sizes()[2] * scale_w),
            self.sizes()[3]};
      }
    }
  }

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
      self.sizes()[2], out_shape.at(2), self.sizes()[3], out_shape.at(3));
  return {out_shape};
}
// Backward Output Shape - Bicubic2D
sizes_vec UpsampleBicubic2DBwdOutputShape(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(4);
  std::vector<int64_t> outshape = stack.at(2).toIntVector();
  CHECK_NULL_INPUT(out_size, scale);
  upsample_2d_common_check(grad_in, out_size, scale);
  if (!habana_lazy::isDeviceInLoweringMode() ||
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == true) {
    return {outshape};
  } else { // always in NHWC - backend
    return {{outshape[0], outshape[2], outshape[3], outshape[1]}};
  }
}
// Forward Output Shape - Nearest3D
sizes_vec UpsampleNearest3DFwdOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(2);
  std::vector<int64_t> out_shape;
  upsample_3d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);
  if (!out_size.isNone()) {
    out_shape = {
        self.sizes()[0],
        self.sizes()[1],
        out_size.toIntVector().at(0),
        out_size.toIntVector().at(1),
        out_size.toIntVector().at(2)};
  } else if (!scale.isNone()) {
    double scale_d = scale.toDoubleVector().at(0);
    double scale_h = scale.toDoubleVector().at(1);
    double scale_w = scale.toDoubleVector().at(2);
    out_shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(self.sizes()[2] * scale_d),
        static_cast<int64_t>(self.sizes()[3] * scale_h),
        static_cast<int64_t>(self.sizes()[4] * scale_w)};
  }
  CHECK_INPUT_OUTPUT_DEPTH_HEIGHT_WIDTH(
      self.sizes()[2],
      out_shape.at(2),
      self.sizes()[3],
      out_shape.at(3),
      self.sizes()[4],
      out_shape.at(4));
  return {out_shape};
}
// Backward Output Shape - Nearest3D
sizes_vec UpsampleNearest3DBwdOutputShape(const at::Stack& stack) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  auto outshape = stack.at(2).toIntVector();
  CHECK_NULL_INPUT(out_size, scale);
  upsample_3d_common_check(grad_in, out_size, scale);
  return {outshape};
}

enum modes { nearest, linear, bicubic };

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
  params->nearestMode = ResizeNearestMode_t::FLOOR;
  if (upsample_mode == nearest) {
    params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
  } else if (upsample_mode == linear) {
    params->mode = ResizeInterpolationMode_t::RESIZE_INTER_LINEAR;
  } else if (upsample_mode == bicubic) {
    params->mode = ResizeInterpolationMode_t::RESIZE_INTER_CUBIC;
    params->nearestMode = ResizeNearestMode_t::ROUND_DEFAULT;
    params->cubicCoeffA =
        -0.75; // As mentioned in TPC guide, value of cubicCoeffA used for cubic
               // interpolation is -0.75.
  }
  params->excludeOutside = false;
  if (upsample_mode != nearest) {
    params->coordTransMode = align_corner
        ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
        : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
  } else {
    params->coordTransMode =
        ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
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
// Transpose NCW/NCHW/NCDHW to NWC/NDHWC and vice versa
static std::vector<synapse_helpers::tensor> Transpose_MemFormat(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const int variant_type,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    bool persistant,
    c10::optional<int> final_index = c10::nullopt) {
  synTransposeParams trans_params{};
  trans_params.tensorDim = variant_type;
  for (int i = 0; i < (variant_type); ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  if (variant_type == 3) { // 1D variant
    std::swap(trans_params.permutation[0], trans_params.permutation[1]);
  } else if (variant_type == 5 && !persistant) { // 3D variant Fwd
    std::swap(trans_params.permutation[2], trans_params.permutation[3]);
    std::swap(trans_params.permutation[1], trans_params.permutation[2]);
    std::swap(trans_params.permutation[0], trans_params.permutation[1]);
  } else if (variant_type == 5 && persistant) { // 3D variant Bwd
    std::swap(trans_params.permutation[0], trans_params.permutation[1]);
    std::swap(trans_params.permutation[1], trans_params.permutation[2]);
    std::swap(trans_params.permutation[2], trans_params.permutation[3]);
  }
  return OpBackend::BuildNode(
      op,
      graph,
      {"transpose",
       std::move(input),
       {{outshape, op->ScalarType(), final_index}},
       &trans_params,
       sizeof(trans_params)});
}
// Resize TPC kernel
static std::vector<synapse_helpers::tensor> Resize(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_index = c10::nullopt) {
  return OpBackend::BuildNode(
      op,
      graph,
      {op->GetGuid(),
       std::move(input),
       {{outshape, op->ScalarType(), final_index}},
       params.get(),
       size});
}
// Slice the result when both scale and size provided - outplace fwd varaint
static std::vector<synapse_helpers::tensor> Slice(
    OpBackend* op,
    synapse_helpers::graph& graph,
    int input_size,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    c10::optional<int> final_index = c10::nullopt) {
  if (input_size == 3) {
    // 3D inputs are reshaped to 4D inputs
    input_size = 4;
  }
  synSliceParamsNDims slice_params{};
  for (int i = input_size - 1; i >= 0; --i) {
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
       {{outshape, op->ScalarType(), final_index}},
       &slice_params,
       sizeof(slice_params)});
}

// Upsample Common function - Old Layout
std::vector<synapse_helpers::tensor> UpsampleCommonFuncOldLayout(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes upsample_mode,
    bool isForward,
    std::vector<synTensor> input,
    const at::IntArrayRef shape_in,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    const at::IntArrayRef outshape,
    const int variant_type) {
  std::vector<int64_t> out_shape_temp;
  // Transpose MemLayout
  if (variant_type == 3) { // 1D - N,C,W to N,W,C
    out_shape_temp = {shape_in[0], shape_in[2], shape_in[1]};
  } else if (variant_type == 5) { // 3D - N,C,D,H,W to N,D,H,W,C
    out_shape_temp = {
        shape_in[0], shape_in[2], shape_in[3], shape_in[4], shape_in[1]};
  }
  auto transpose = Transpose_MemFormat(
      op, graph, variant_type, std::move(input), out_shape_temp, false);

  // Reshape - 1D varaints only
  // N,W,C to N,H,W,C where H=1
  if (variant_type == 3) {
    out_shape_temp = {
        shape_in[0], static_cast<int64_t>(1), shape_in[2], shape_in[1]};
    transpose.front() = OpBackend::BuildReshape(
        op, graph, transpose[0].get(), out_shape_temp, op->ScalarType());
  }
  // Resize
  // modify input width value with output width value
  // when both size and scale is provided with align_corners=false
  if (isForward && !align_corners && (!out_size.isNone() && !scales.isNone())) {
    if (variant_type == 3) { // 1D
      out_shape_temp.at(2) = static_cast<int64_t>(shape_in[2] * scale_w);
    } else if (variant_type == 5) { // 3D
      out_shape_temp.at(1) = static_cast<int64_t>(shape_in[2] * scale_d);
      out_shape_temp.at(2) = static_cast<int64_t>(shape_in[3] * scale_h);
      out_shape_temp.at(3) = static_cast<int64_t>(shape_in[4] * scale_w);
    }
  } else {
    if (variant_type == 3) { // 1D
      out_shape_temp.at(2) = outshape.at(2);
    } else if (variant_type == 5) { // 3D
      out_shape_temp.at(1) = outshape.at(2);
      out_shape_temp.at(2) = outshape.at(3);
      out_shape_temp.at(3) = outshape.at(4);
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
  auto resize =
      Resize(op, graph, {transpose[0].get()}, out_shape_temp, params, size);
  // Slice
  // For Fwd ops, when both size and scale is provided with align_corners=false
  if (isForward && !align_corners && (!out_size.isNone() && !scales.isNone())) {
    std::vector<int64_t> slice_shape;
    if (variant_type == 3) { // 1D
      slice_shape = {shape_in[0], 1 /*H*/, outshape.at(2), shape_in[1]};
    } else if (variant_type == 5) { // 3D
      slice_shape = {
          shape_in[0],
          outshape.at(2),
          outshape.at(3),
          outshape.at(4),
          shape_in[1]};
    }
    resize = Slice(op, graph, variant_type, {resize[0].get()}, slice_shape);
  };

  // Reshape - 1D variants only
  // N,H,W,C to N,W,C where H=1
  if (variant_type == 3) {
    std::vector<int64_t> out_shape_3d = {
        shape_in[0], outshape.at(2), shape_in[1]};
    resize.front() = OpBackend::BuildReshape(
        op, graph, resize[0].get(), out_shape_3d, op->ScalarType());
  }
  // Transpose to Pytorch MemLayout
  return Transpose_MemFormat(
      op, graph, variant_type, {resize[0].get()}, outshape, true, 0);
}

// Upsample Common function - New Layout
std::vector<synapse_helpers::tensor> UpsampleCommonFuncSynapseLayout(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes upsample_mode,
    bool isForward,
    std::vector<synTensor> input,
    const at::IntArrayRef shape_in,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    const at::IntArrayRef outshape,
    const int variant_type) {
  std::vector<int64_t> out_shape_temp(shape_in.begin(), shape_in.end());
  std::vector<synTensor> reshaped_input(std::move(input));
  std::vector<synapse_helpers::tensor> reshape;

  // Reshape - 1D varaints only
  // N,C,W to N,C,H,W where H=2
  if (variant_type == 3) {
    out_shape_temp = {
        shape_in[0], shape_in[1], static_cast<int64_t>(1), shape_in[2]};
    reshape.emplace_back(OpBackend::BuildReshape(
        op, graph, reshaped_input[0], out_shape_temp, op->ScalarType()));
    reshaped_input[0] = reshape[0].get();
  }
  // Resize
  // modify input width value with output width value
  // when both size and scale is provided with align_corners=false
  bool modifyInputWithOutputWidth =
      isForward && !align_corners && (!out_size.isNone() && !scales.isNone());
  if (modifyInputWithOutputWidth) {
    if (variant_type == 3) { // 1D
      out_shape_temp.at(3) = static_cast<int64_t>(shape_in[2] * scale_w);
    } else if (variant_type == 5) { // 3D
      out_shape_temp.at(2) = static_cast<int64_t>(shape_in[2] * scale_d);
      out_shape_temp.at(3) = static_cast<int64_t>(shape_in[3] * scale_h);
      out_shape_temp.at(4) = static_cast<int64_t>(shape_in[4] * scale_w);
    }
  } else {
    if (variant_type == 3) { // 1D
      out_shape_temp.at(3) = outshape.at(2);
    } else if (variant_type == 5) { // 3D
      out_shape_temp.at(1) = outshape.at(1);
      out_shape_temp.at(2) = outshape.at(2);
      out_shape_temp.at(3) = outshape.at(3);
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
  auto final_index_for_resize = modifyInputWithOutputWidth || variant_type == 3
      ? c10::optional<int>()
      : c10::optional<int>(0);
  auto resize = Resize(
      op,
      graph,
      reshaped_input,
      out_shape_temp,
      params,
      size,
      final_index_for_resize);
  // Slice
  // For Fwd ops, when both size and scale is provided with align_corners=false
  if (modifyInputWithOutputWidth) {
    std::vector<int64_t> slice_shape(outshape.begin(), outshape.end());
    if (variant_type == 3) { // 1D
      // NCHW
      slice_shape = {shape_in[0], shape_in[1], 1 /*H*/, outshape.at(2)};
    }
    auto final_index_for_slice =
        variant_type == 3 ? c10::optional<int>() : c10::optional<int>(0);
    resize = Slice(
        op,
        graph,
        variant_type,
        {resize[0].get()},
        slice_shape,
        final_index_for_slice);
  };

  // Reshape - 1D variants only
  // N,C,H,W to N,C,W where H=2
  if (variant_type == 3) {
    resize.front() = OpBackend::BuildReshape(
        op, graph, resize[0].get(), outshape, op->ScalarType(), 0);
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
    const at::IntArrayRef shape_in,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    const at::IntArrayRef outshape,
    const int variant_type) {
  PT_LAZY_DEBUG(__FUNCTION__);
  std::vector<synapse_helpers::tensor> output;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    output = UpsampleCommonFuncSynapseLayout(
        op,
        graph,
        upsample_mode,
        isForward,
        input,
        shape_in,
        out_size,
        align_corners,
        scales,
        scale_w,
        scale_h,
        scale_d,
        outshape,
        variant_type);
  }
  //! PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING
  else {
    output = UpsampleCommonFuncOldLayout(
        op,
        graph,
        upsample_mode,
        isForward,
        input,
        shape_in,
        out_size,
        align_corners,
        scales,
        scale_w,
        scale_h,
        scale_d,
        outshape,
        variant_type);
  }
  return output;
}

// AddNode FWD 1D Linear function
void UpsampleLinear1DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = UpsampleLinear1DFwdOutputShape(stack)[0];
  auto self = stack.at(0).toTensor();
  auto shape_in = self.sizes();
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
      shape_in,
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      output_shape,
      self.dim() /*variant_type - 1D*/
  );

  syn_out(0) = std::move(result.at(0));
}
// AddNode BWD 1D Linear function
void UpsampleLinear1DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = UpsampleLinear1DBwdOutputShape(stack)[0];
  auto self = stack.at(0).toTensor();
  auto shape_in = self.sizes();
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
      shape_in,
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      output_shape,
      self.dim() /*variant_type - 1D*/
  );
  syn_out(0) = std::move(result.at(0));
}
// AddNode FWD 1D Nearest function
void UpsampleNearest1DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = UpsampleNearest1DFwdOutputShape(stack)[0];
  auto self = stack.at(0).toTensor();
  auto shape_in = self.sizes();
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
      shape_in,
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      output_shape,
      self.dim() /*variant_type - 1D*/
  );
  syn_out(0) = std::move(result.at(0));
}
// AddNode BWD 1D Nearest function
void UpsampleNearest1DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = UpsampleNearest1DBwdOutputShape(stack)[0];
  auto self = stack.at(0).toTensor();
  auto shape_in = self.sizes();
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
      shape_in,
      out_size,
      align_corners,
      scales,
      scale_w,
      1.0 /*scale_h*/,
      1.0 /*scale_d*/,
      output_shape,
      self.dim() /*variant_type - 1D*/
  );
  syn_out(0) = std::move(result.at(0));
}
// AddNode FWD 3D Nearest function
void UpSampleNearest3DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = UpsampleNearest3DFwdOutputShape(stack)[0];
  auto self = stack.at(0).toTensor();
  auto shape_in = self.sizes();
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
  auto result = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      shape_in,
      out_size,
      false, /*align_corners*/
      scales,
      scale_w,
      scale_h,
      scale_d,
      output_shape,
      self.dim() /*variant_type - 3D*/
  );
  syn_out(0) = std::move(result.at(0));
}
// AddNode BWD 3D Nearest function
void UpSampleNearest3DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // outshape
  auto output_shape = UpsampleNearest3DBwdOutputShape(stack)[0];
  auto self = stack.at(0).toTensor();
  auto shape_in = self.sizes();
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
  auto result = UpsampleCommonFunc(
      this,
      graph,
      nearest, /*upsample_mode*/
      false, /*isForward*/
      {syn_in(0)},
      shape_in,
      out_size,
      false, /*align_corners*/
      scales,
      scale_w,
      scale_h,
      scale_d,
      output_shape,
      self.dim() /*variant_type - 3D*/
  );
  syn_out(0) = std::move(result.at(0));
}
} // namespace habana
