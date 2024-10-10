/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#pragma once

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

} // namespace habana
