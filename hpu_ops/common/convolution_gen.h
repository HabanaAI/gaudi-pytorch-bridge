/**
* Copyright (c) 2021-2024 Intel Corporation
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
#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <vector>

namespace habana {

#define IF_CONV1D_EXPAND_TO_2D(at_input, input_idx)                   \
  synTensor at_input##_expanded = syn_in(input_idx);                  \
  std::optional<synapse_helpers::tensor> at_input##_expanded_storage; \
  if (is_conv_1d) {                                                   \
    std::vector<int64_t> sizes_4d = at_input.sizes().vec();           \
    sizes_4d.push_back(1);                                            \
                                                                      \
    synAxisParams expandParams{0};                                    \
    at_input##_expanded_storage = std::move(BuildOp(                  \
        graph,                                                        \
        "expand_dims",                                                \
        {syn_in(input_idx)},                                          \
        {{sizes_4d, at_input.scalar_type()}},                         \
        &expandParams,                                                \
        sizeof(expandParams))[0]);                                    \
    at_input##_expanded = (*at_input##_expanded_storage).get();       \
  }

#define IF_CONV1D_SQUEEZE_TO_ORIG_AND_SET_OUT(                 \
    out, out_shape, final_result_index)                        \
  if (is_conv_1d) {                                            \
    std::vector<int64_t> output_sizes_3d = out_shape;          \
    output_sizes_3d.pop_back();                                \
                                                               \
    synAxisParams squeezeParams{0};                            \
    auto output_squeezed = std::move(BuildOp(                  \
        graph,                                                 \
        "squeeze",                                             \
        {(out).get()},                                         \
        {{output_sizes_3d, ScalarType(), final_result_index}}, \
        &squeezeParams,                                        \
        sizeof(squeezeParams))[0]);                            \
                                                               \
    syn_out(final_result_index) = std::move(output_squeezed);  \
  } else {                                                     \
    syn_out(final_result_index) = std::move(out);              \
  }

// It was previously done by the PT, but since we moved from
// convolution_overrideable to convolution, we have to do it.
std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim);

#define FRONTEND_CONVOLUTION_COMMON(shift)                             \
  auto weight = inputs[1 + shift].toTensor();                          \
  const auto params_dim = weight.dim() - 2;                            \
  auto& pt_inputs = get_inputs();                                      \
                                                                       \
  pt_inputs[3 + shift] = expand_param_if_needed(                       \
      pt_inputs[3 + shift].toIntList().vec(), "stride", params_dim);   \
  pt_inputs[4 + shift] = expand_param_if_needed(                       \
      pt_inputs[4 + shift].toIntList().vec(), "padding", params_dim);  \
  pt_inputs[5 + shift] = expand_param_if_needed(                       \
      pt_inputs[5 + shift].toIntList().vec(), "dilation", params_dim); \
  pt_inputs[7 + shift] = expand_param_if_needed(                       \
      pt_inputs[7 + shift].toIntList().vec(), "outputPadding", params_dim);

} // namespace habana
