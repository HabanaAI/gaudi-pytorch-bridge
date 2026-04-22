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

#include "generated/backend/im2col.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

// SW-215089
// Only this specific configuration is supported by TPC for now.
bool Im2ColFallbackCheck(
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  return (
      kernel_size[0] == 14 && kernel_size[1] == 14 && dilation[0] == 1 &&
      dilation[1] == 1 && padding[0] == 0 && padding[1] == 0 &&
      stride[0] == 14 && stride[1] == 14);
}

OutputMetaDataVector Im2ColMeta(const at::Stack& stack) {
  auto input = stack.at(0).toTensor();
  auto kernel_size = stack.at(1).toIntVector();
  auto dilation = stack.at(2).toIntVector();
  auto padding = stack.at(3).toIntVector();
  auto stride = stack.at(4).toIntVector();

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = ((input_height + 2 * padding[0] -
                            (dilation[0] * (kernel_size[0] - 1) + 1)) /
                           stride[0]) +
      1;
  int64_t output_width = ((input_width + 2 * padding[1] -
                           (dilation[1] * (kernel_size[1] - 1) + 1)) /
                          stride[1]) +
      1;
  int64_t n_output_plane = n_input_plane * kernel_size[0] * kernel_size[1];
  int64_t output_length = output_height * output_width;

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = {batch_size, n_output_plane, output_length};
  meta.dtype = input.scalar_type();
  return metaVec;
}

FillParamsT FillIm2ColParams(const at::Stack& stack) {
  auto kernel_size = stack.at(1).toIntVector();
  auto dilation = stack.at(2).toIntVector();
  auto padding = stack.at(3).toIntVector();
  auto stride = stack.at(4).toIntVector();

  PARAMS_STUB(ns_Im2Col::Params);
  check_range<int>(0, 1, kernel_size);
  params->kernel_h = static_cast<int>(kernel_size[0]);
  params->kernel_w = static_cast<int>(kernel_size[1]);
  check_range<int>(0, 1, dilation);
  params->dilation_h = static_cast<int>(dilation[0]);
  params->dilation_w = static_cast<int>(dilation[1]);
  check_range<int>(0, 1, padding);
  params->pad_h = static_cast<int>(padding[0]);
  params->pad_w = static_cast<int>(padding[1]);
  check_range<int>(0, 1, stride);
  params->stride_h = static_cast<int>(stride[0]);
  params->stride_w = static_cast<int>(stride[1]);
  return paramsT;
}

} // namespace habana
