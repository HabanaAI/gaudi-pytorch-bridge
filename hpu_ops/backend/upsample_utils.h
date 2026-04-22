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

#include <torch/torch.h>
#include "backend/habana_operator.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/fillparams.h"

namespace habana {

class OpBackend;

namespace upsample_utils {

enum modes { nearest, nearest_exact, linear, bicubic };

void upsample_1d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales);

void upsample_2d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales);

void upsample_exact_2d_check(const torch::Tensor& input, c10::IValue out_size);

void upsample_3d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales);

void upsample_exact_3d_check(const torch::Tensor& input, c10::IValue out_size);

std::vector<synapse_helpers::tensor> Resize(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    const at::ScalarType& dtype,
    const FillParamsT& params,
    std::optional<int> final_index = at::nullopt);

FillParamsT FillResizeParams(
    const int shape_in_dim,
    enum modes upsample_mode,
    c10::IValue out_size,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    bool align_corner,
    bool antialias);

FillParamsT FillResizeParams(
    enum modes upsample_mode,
    std::optional<std::vector<int64_t>> out_size,
    std::optional<std::array<double, 3>> scales,
    bool align_corner,
    bool antialias);

SharedMetaDataVector UpsampleCommmonSharedLayer(
    const at::Stack& stack,
    const bool alignCorners,
    const int64_t scalesIndex,
    const bool isForward,
    const modes upsample_mode,
    const habana_helpers::HabanaExecutionMode execution_mode);

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
    const at::Tensor self_tensor);

inline void check_null_input(c10::IValue out_size, c10::IValue scale) {
  HABANA_ASSERT(
      not out_size.isNone() || not scale.isNone(),
      "Upsample: Must specify exactly one of output_size and scale_factors");
}

inline void check_null_inputs_2d(
    c10::IValue out_size,
    std::optional<double> scale_h,
    std::optional<double> scale_w) {
  HABANA_ASSERT(
      (scale_h.has_value() && scale_w.has_value()) || (not out_size.isNone()),
      "Upsample: Must specify output size if scales aren't given, but got output_size: ",
      out_size,
      " and scale_factors: ",
      scale_h,
      ", ",
      scale_w);
}

inline void check_null_inputs_3d(
    c10::IValue out_size,
    std::optional<double> scale_d,
    std::optional<double> scale_h,
    std::optional<double> scale_w) {
  HABANA_ASSERT(
      (scale_d.has_value() && scale_h.has_value() && scale_w.has_value()) ||
          (not out_size.isNone()),
      "Upsample: Must specify output size if scales aren't given, but got output_size: ",
      out_size,
      " and scale_factors: ",
      scale_d,
      ", ",
      scale_h,
      ", ",
      scale_w);
}

inline void check_input_output_width(
    int64_t input_width,
    int64_t output_width) {
  HABANA_ASSERT(
      input_width > 0 && output_width > 0,
      "Upsample1D: Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");
}

inline void check_input_output_height_width(
    int64_t input_height,
    int64_t output_height,
    int64_t input_width,
    int64_t output_width) {
  HABANA_ASSERT(
      (input_width > 0 && output_width > 0) &&
          (input_height > 0 && output_height > 0),
      "Upsample2D: Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and (H: ",
      input_height,
      ") output (W: ",
      output_width,
      ") for Upsample2D");
}

inline void check_input_output_depth_height_width(
    int64_t input_depth,
    int64_t output_depth,
    int64_t input_height,
    int64_t output_height,
    int64_t input_width,
    int64_t output_width) {
  HABANA_ASSERT(
      (input_depth > 0 && output_depth > 0) &&
          (input_width > 0 && output_width > 0) &&
          (input_height > 0 && output_height > 0),
      "Upsample3D: Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and (H: ",
      input_height,
      ") and (D: ",
      input_depth,
      ") output (W: ",
      output_width,
      ") and (H: ",
      output_height,
      ") and (D: ",
      output_depth,
      ") for Upsample3D");
}

} // namespace upsample_utils

} // namespace habana
