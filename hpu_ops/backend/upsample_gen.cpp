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
#include "generated/backend/upsample_trilinear3d.h"

using namespace synapse_helpers::layouts;

#define CHECK_NULL_INPUT(out_size, scale)                     \
  HABANA_ASSERT(                                              \
      !(out_size == c10::nullopt && scale == c10::nullopt) || \
          (out_size != c10::nullopt &&                        \
           (scale != c10::nullopt && !scale.isScalar())),     \
      "Upsample: Must specify exactly one of output_size and scale_factors");

inline void check_null_inputs_2d(
    c10::IValue out_size,
    c10::optional<double> scale_h,
    c10::optional<double> scale_w) {
  HABANA_ASSERT(
      (scale_h.has_value() && scale_w.has_value()) || !out_size.isNone(),
      "Upsample: Must specify output size if scales aren't given, but got output_size: ",
      out_size,
      " and scale_factors: ",
      scale_h,
      ", ",
      scale_w);
}

inline void check_null_inputs_3d(
    c10::IValue out_size,
    c10::optional<double> scale_d,
    c10::optional<double> scale_h,
    c10::optional<double> scale_w) {
  HABANA_ASSERT(
      (scale_d.has_value() && scale_h.has_value() && scale_w.has_value()) ||
          !out_size.isNone(),
      "Upsample: Must specify output size if scales aren't given, but got output_size: ",
      out_size,
      " and scale_factors: ",
      scale_d,
      ", ",
      scale_h,
      ", ",
      scale_w);
}

#define CHECK_INPUT_OUTPUT_WIDTH(input_width, output_width)                               \
  HABANA_ASSERT(                                                                          \
      input_width > 0 && output_width > 0,                                                \
      "Upsample1D:  Input and output sizes should be greater than 0, but got input (W: ", \
      input_width,                                                                        \
      ") and output (W: ",                                                                \
      output_width,                                                                       \
      ")");

#define CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(                                                  \
    input_height, output_height, input_width, output_width)                               \
  HABANA_ASSERT(                                                                          \
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
  HABANA_ASSERT(                                                                         \
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
  HABANA_ASSERT(
      input.dim() == 3,
      "Upsample1D expects input_size equals to 3, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 1,
        "Upsample1D expects out_size equals to 1, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    HABANA_ASSERT(
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
  HABANA_ASSERT(
      input.dim() == 4,
      "Upsample2D expects input_size equals to 4, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 2,
        "Upsample2D expects out_size equals to 2, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    HABANA_ASSERT(
        scales.toDoubleVector().size() == 2,
        "Upsample2D expects scales equals to 2, but got ",
        scales.toDoubleVector().size());
  }
}

void upsample_exact_2d_check(const torch::Tensor& input, c10::IValue out_size) {
  HABANA_ASSERT(
      input.dim() == 4,
      "Upsample2D expects input_size equals to 4, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 2,
        "Upsample2D expects out_size equals to 2, but got ",
        out_size.toIntVector().size());
  }
}

void upsample_3d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales) {
  HABANA_ASSERT(
      input.dim() == 5,
      "Upsample3D expects input_size equals to 5, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 3,
        "Upsample3D expects out_size equals to 3, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    HABANA_ASSERT(
        scales.toDoubleVector().size() == 3,
        "Upsample3D expects scales equals to 3, but got ",
        scales.toDoubleVector().size());
  }
}

void upsample_exact_3d_check(const torch::Tensor& input, c10::IValue out_size) {
  HABANA_ASSERT(
      input.dim() == 5,
      "Upsample3D expects input_size equals to 5, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 3,
        "Upsample3D expects out_size equals to 3, but got ",
        out_size.toIntVector().size());
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
        static_cast<int64_t>(self_sizes.at(INPUT_H_IDX) * scale_h),
        static_cast<int64_t>(self_sizes.at(INPUT_W_IDX) * scale_w)};
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
        static_cast<int64_t>(self_sizes.at(INPUT_C_IDX) * scale_d),
        static_cast<int64_t>(self_sizes.at(INPUT_H_IDX) * scale_h),
        static_cast<int64_t>(self_sizes.at(INPUT_W_IDX) * scale_w)};
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

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
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

  CHECK_INPUT_OUTPUT_DEPTH_HEIGHT_WIDTH(
      self.sizes()[2],
      meta.shape.at(2),
      self.sizes()[3],
      meta.shape.at(3),
      self.sizes()[4],
      meta.shape.at(4));
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
  upsample_2d_common_check(self, out_size, scale);
  CHECK_NULL_INPUT(out_size, scale);

  OutputMetaData meta;
  meta.shape = UpsampleBicubic2DFwdOutputShapeSynapseLayout(stack);
  meta.dtype = self.scalar_type();

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
      self.sizes()[2], meta.shape.at(2), self.sizes()[3], meta.shape.at(3));

  return {meta};
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

  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
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

enum modes { nearest, nearest_exact, linear, bicubic };

SharedMetaDataVector UpsampleCommmonSharedLayer(
    const at::Stack& stack,
    const bool alignCorners,
    const int64_t scalesIndex,
    const bool isForward) {
  const auto& self = stack_tensor(stack, 0);
  const auto& outSize = stack.at(1);
  const auto& scales = stack.at(scalesIndex);
  const bool modifyInputWithOutputWidth =
      isForward && !alignCorners && (!outSize.isNone() && !scales.isNone());

  SharedMetaDataVector metaVec;
  metaVec.reserve(modifyInputWithOutputWidth ? 2 : 1);

  const auto rank = self.dim();
  auto dtype = self.scalar_type();
  if (dtype == c10::ScalarType::Byte)
    dtype = c10::ScalarType::Float;

  const std::string guid = isForward ? "resize_fwd" : "resize_bwd";
  SharedMetaTensor commonTensor = {rank, dtype};
  SharedMetaData resizeSharedMeta{guid};
  resizeSharedMeta.inputs_data = {commonTensor};
  resizeSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(resizeSharedMeta);

  if (modifyInputWithOutputWidth) {
    SharedMetaData sliceSharedMeta{"slice"};
    sliceSharedMeta.inputs_data = {commonTensor};
    sliceSharedMeta.outputs_data = {commonTensor};
    metaVec.push_back(sliceSharedMeta);
  }

  return metaVec;
}

SharedMetaDataVector UpsampleLinear1DFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return UpsampleCommmonSharedLayer(stack, stack.at(2).toBool(), 3, true);
}

SharedMetaDataVector UpsampleLinear1DBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return UpsampleCommmonSharedLayer(stack, stack.at(3).toBool(), 4, false);
}

SharedMetaDataVector UpsampleNearest1D3DFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return UpsampleCommmonSharedLayer(stack, false, 2, true);
}

SharedMetaDataVector UpsampleNearest1D3DBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return UpsampleCommmonSharedLayer(stack, false, 3, false);
}

SharedMetaDataVector UpsampleNearest2DFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return UpsampleCommmonSharedLayer(stack, true, 2, true);
}

SharedMetaDataVector UpsampleNearest2DBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return UpsampleCommmonSharedLayer(stack, true, 3, false);
}

SharedMetaDataVector UpssampleTrilinear3DSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return UpsampleCommmonSharedLayer(stack, stack.at(2).toBool(), 3, true);
}

// Custom FillParams function
std::shared_ptr<void> FillResizeParams(
    const int shape_in_dim,
    size_t& size,
    enum modes upsample_mode,
    c10::IValue out_size,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    bool align_corner,
    bool antialias) {
  PARAMS_STUB(ns_ResizeKernel::ParamsAA);
  params->excludeOutside = false;
  params->useAntialiasing = antialias;
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
    if (shape_in_dim == 3) { // 1D variant
      params->size1 = out_size.toIntVector().at(0);
    } else if (shape_in_dim == 4) { // 2D variant
      params->size1 = out_size.toIntVector().at(1);
      params->size2 = out_size.toIntVector().at(0);
    } else if (shape_in_dim == 5) { // 3D variant
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
      align_corners,
      false /*antialias*/);
}

std::shared_ptr<void> FillBicubicFwdParamsAA(
    const at::Stack& stack,
    size_t& size) {
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
      size,
      bicubic,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

std::shared_ptr<void> FillBicubicBwdParamsAA(
    const at::Stack& stack,
    size_t& size) {
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
      size,
      bicubic,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
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
      align_corners,
      false /*antialias*/);
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
      align_corners,
      false /*antialias*/);
}

std::tuple<double, double, double> ExtractScales(
    const at::IValue& scales,
    const at::Stack& stack,
    size_t scale_h_idx,
    size_t scale_w_idx) {
  double scale_w = 1.0, scale_h = 1.0, scale_d = 1.0;
  if (!scales.isNone()) {
    scale_h = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : stack.at(scale_h_idx).toDouble();
    scale_w = !scales.isScalar() ? scales.toDoubleVector().at(1)
                                 : stack.at(scale_w_idx).toDouble();
  }
  return {scale_w, scale_h, scale_d};
}

std::shared_ptr<void> FillBilinearParamsAAHelper(
    const at::Tensor& input_tensor,
    const at::Stack& stack,
    const at::IValue& out_size,
    const at::IValue& scales,
    bool align_corners,
    size_t& size,
    size_t scale_h_idx,
    size_t scale_w_idx) {
  auto [scale_w, scale_h, scale_d] =
      ExtractScales(scales, stack, scale_h_idx, scale_w_idx);
  return FillResizeParams(
      input_tensor.dim(),
      size,
      linear,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      true /*antialias*/);
}

std::shared_ptr<void> FillBilinearFwdParamsAA(
    const at::Stack& stack,
    size_t& size) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(2).toBool();
  auto scales = stack.at(3);
  return FillBilinearParamsAAHelper(
      self, stack, out_size, scales, align_corners, size, 3, 4);
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
      align_corners,
      false /*antialias*/);
}

std::shared_ptr<void> FillBilinearBwdParamsAA(
    const at::Stack& stack,
    size_t& size) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto align_corners = stack.at(3).toBool();
  auto scales = stack.at(4);
  return FillBilinearParamsAAHelper(
      grad_in, stack, out_size, scales, align_corners, size, 4, 5);
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
      false /*align_corners*/,
      false /*antialias*/);
}

std::shared_ptr<void> FillNearestExact2DFwdParams(
    const at::Stack& stack,
    size_t& size) {
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
      size,
      nearest_exact,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

std::shared_ptr<void> FillNearestExact2DBwdParams(
    const at::Stack& stack,
    size_t& size) {
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
      size,
      nearest_exact,
      out_size,
      scales_h,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
}

std::shared_ptr<void> FillNearestExact3DFwdParams(
    const at::Stack& stack,
    size_t& size) {
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
      size,
      nearest_exact,
      out_size,
      scales,
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
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
      false /*align_corners*/,
      false /*antialias*/);
}

std::shared_ptr<void> FillNearestExact3DBwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto grad_in = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  bool align_corners = false;
  bool antialias = false;
  double scale_d = stack.at(3).toOptional<double>().value_or(1.0);
  double scale_h = stack.at(4).toOptional<double>().value_or(1.0);
  double scale_w = stack.at(5).toOptional<double>().value_or(1.0);

  return FillResizeParams(
      grad_in.dim(),
      size,
      nearest_exact,
      out_size,
      stack.at(3),
      scale_w,
      scale_h,
      scale_d,
      align_corners,
      antialias);
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
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    const at::ScalarType& dtype,
    c10::optional<int> final_index = c10::nullopt) {
  auto output_size = outshape.size();

  synSliceParamsV2 slice_params{};
  for (int64_t i = output_size - 1; i >= 0; --i) {
    slice_params.axes[i] = i;
    slice_params.starts[i] = 0;
    slice_params.ends[i] = outshape[(output_size - i - 1)];
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
synapse_helpers::tensor UpsampleCommonFuncSynapseLayout(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes upsample_mode,
    bool isForward,
    std::vector<synTensor>&& input,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    const std::array<double, 3>& scale_dhw,
    const OutputMetaData& meta,
    const at::Tensor self_tensor) {
  auto shape_in_dim = self_tensor.dim();
  auto shape_in = self_tensor.sizes();
  const std::vector<int64_t>* p_shape_out_resize = &meta.shape;

  std::optional<synapse_helpers::tensor> cast_storage;
  auto intermediateDtype = meta.dtype;
  if (meta.dtype == c10::ScalarType::Byte) {
    // u8 to f32
    cast_storage = OpBackend::BuildCast(
        op,
        graph,
        input[0],
        shape_in,
        c10::ScalarType::Byte,
        c10::ScalarType::Float);
    input[0] = cast_storage->get();
    intermediateDtype = c10::ScalarType::Float;
  }
  // Resize
  // modify input width value with output width value
  // when both size and scale is provided with align_corners=false
  bool modifyInputWithOutputWidth =
      isForward && !align_corners && (!out_size.isNone() && !scales.isNone());
  std::vector<int64_t> shape_out_resize;
  if (modifyInputWithOutputWidth) {
    shape_out_resize.reserve(shape_in_dim);
    unsigned scaled_dims = (shape_in_dim > 2) ? shape_in_dim - 2 : 0;

    for (unsigned d = 0; d < shape_in_dim - scaled_dims; ++d)
      shape_out_resize.push_back(meta.shape[d]);

    for (unsigned d = 0; d < scaled_dims; ++d)
      shape_out_resize.push_back(static_cast<int64_t>(
          shape_in[2 + d] * scale_dhw[d + 3 - scaled_dims]));

    p_shape_out_resize = &shape_out_resize;
  }

  size_t size = 0;
  const auto& params = FillResizeParams(
      shape_in_dim,
      size,
      upsample_mode,
      out_size,
      scales,
      scale_dhw[2],
      scale_dhw[1],
      scale_dhw[0],
      align_corners,
      false /*antialias*/);
  auto final_index_for_resize =
      modifyInputWithOutputWidth || meta.dtype == c10::ScalarType::Byte
      ? c10::optional<int>()
      : c10::optional<int>(0);

  auto resize = Resize(
      op,
      graph,
      input,
      *p_shape_out_resize,
      intermediateDtype,
      params,
      size,
      final_index_for_resize);
  // Slice
  // For Fwd ops, when both size and scale is provided with align_corners=false
  if (modifyInputWithOutputWidth) {
    auto final_index_for_slice = (meta.dtype == c10::ScalarType::Byte)
        ? c10::optional<int>()
        : c10::optional<int>(0);

    resize = Slice(
        op,
        graph,
        {resize[0].get()},
        meta.shape,
        intermediateDtype,
        final_index_for_slice);
  };
  if (meta.dtype != c10::ScalarType::Byte)
    return std::move(resize[0]);

  // f32 to u8
  return OpBackend::BuildCast(
      op, graph, resize[0].get(), meta.shape, intermediateDtype, meta.dtype, 0);
}

// Upsample Common function
synapse_helpers::tensor UpsampleCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes upsample_mode,
    bool isForward,
    std::vector<synTensor> input,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    const std::array<double, 3>& scale_dhw,
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
      std::move(input),
      out_size,
      align_corners,
      scales,
      scale_dhw,
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
  c10::optional<int> final_index = 0;
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

synapse_helpers::tensor UpsampleNearestExactFwdCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<synTensor> input,
    const std::shared_ptr<void>& params,
    size_t size) {
  auto meta = op->OutputMeta(stack)[0];
  auto self = stack_tensor(stack, 0);
  std::optional<synapse_helpers::tensor> cast_storage;
  c10::optional<int> final_index = 0;
  op->CreateShapeTensorInput(
      graph, meta.dtype, meta.shape, input, SHAPE_TENSOR);
  auto intermediateDtype = meta.dtype;
  if (meta.dtype == c10::ScalarType::Byte) {
    // u8 to f32
    intermediateDtype = c10::ScalarType::Float;
    cast_storage = OpBackend::BuildCast(
        op, graph, input[0], self.sizes().vec(), meta.dtype, intermediateDtype);
    input[0] = cast_storage->get();
    final_index = c10::nullopt;
  }

  auto resize = Resize(
      op,
      graph,
      input,
      meta.shape,
      intermediateDtype,
      params,
      size,
      final_index);
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
  size_t size;
  auto params = FillParams(stack, size);
  syn_out(0) = UpsampleNearestExactFwdCommon(
      this, graph, stack, {syn_in(0)}, params, size);
}
// AddNode FWD 3D Nearest Exact function
void UpsampleNearestExact3DFwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size;
  auto params = FillParams(stack, size);
  syn_out(0) = UpsampleNearestExactFwdCommon(
      this, graph, stack, {syn_in(0)}, params, size);
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
  syn_out(0) = UpsampleCommonFunc(
      this,
      graph,
      linear, /*upsample_mode*/
      true, /*isForward*/
      {syn_in(0)},
      out_size,
      align_corners,
      scales,
      {scale_d, scale_h, scale_w},
      meta,
      self_tensor);
}
// AddNode BWD 2D Nearest Exact function
void UpsampleNearestExact2DBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = UpsampleNearestExact2DBwdMeta(stack)[0];
  c10::optional<int> final_index = 0;

  size_t size = 0;
  const auto& params = FillParams(stack, size);

  auto resize = Resize(
      this,
      graph,
      {syn_in(0)},
      meta.shape,
      meta.dtype,
      params,
      size,
      final_index);

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
  double scale_d = 1.0, scale_w = 1.0, scale_h = 1.0;
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
  double scale_d = 1.0, scale_w = 1.0, scale_h = 1.0;
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
