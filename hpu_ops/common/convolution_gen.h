/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#define IF_CONV1D_RESHAPE_TO_2D(at_input, input_idx)                  \
  synTensor at_input##_reshaped = syn_in(input_idx);                  \
  std::optional<synapse_helpers::tensor> at_input##_reshaped_storage; \
  if (is_conv_1d) {                                                   \
    std::vector<int64_t> sizes_4d = at_input.sizes().vec();           \
    sizes_4d.push_back(1);                                            \
                                                                      \
    at_input##_reshaped_storage = ReshapeHelper(                      \
        graph, syn_in(input_idx), sizes_4d, at_input.scalar_type());  \
    at_input##_reshaped = (*at_input##_reshaped_storage).get();       \
  }

#define IF_CONV1D_RESHAPE_TO_ORIG_AND_SET_OUT(                \
    out, out_shape, final_result_index)                       \
  if (is_conv_1d) {                                           \
    std::vector<int64_t> output_sizes_3d = out_shape;         \
    output_sizes_3d.pop_back();                               \
                                                              \
    auto output_reshaped = ReshapeHelper(                     \
        graph,                                                \
        (out).get(),                                          \
        output_sizes_3d,                                      \
        ScalarType(),                                         \
        final_result_index);                                  \
                                                              \
    syn_out(final_result_index) = std::move(output_reshaped); \
  } else {                                                    \
    syn_out(final_result_index) = std::move(out);             \
  }

} // namespace habana
