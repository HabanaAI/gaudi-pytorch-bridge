
/*******************************************************************************
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
#include "generated/backend/slice_backward.h"

namespace habana {

template <typename idx_t, typename size_t>
idx_t normalize_idx(idx_t idx, size_t size) {
  if (size <= 0) {
    return 0;
  }

  if (idx < -size) {
    idx = 0;
  }

  if (idx > size) {
    idx = size;
  }

  if (idx < 0) {
    idx += size;
  }

  return idx;
}

OutputMetaDataVector SliceBackwardMeta(const at::Stack& stack) {
  auto self = stack[0].toTensor();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = stack[1].toIntList().vec();

  return {meta};
}

void SliceBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "SliceBackward::AddNode");
  auto grad = getNextInput<TensorsPair>(stackGetter);
  auto input_sizes = getNextInput<std::vector<int64_t>>(stackGetter);
  auto dim = getNextInput<int>(stackGetter);
  auto start = getNextInput<int>(stackGetter);
  auto end = getNextInput<int>(stackGetter);
  auto step = getNextInput<int>(stackGetter);

  auto meta = SliceBackwardMeta(stack);

  const auto grad_sizes = meta[0].shape;
  const auto grad_scalar_type = meta[0].dtype;

  if (std::find(grad_sizes.begin(), grad_sizes.end(), 0) != grad_sizes.end()) {
    auto zero_tensor =
        ConstantHelper(graph, 0, grad_scalar_type, input_sizes, 0);
    syn_out(0) = std::move(zero_tensor);
  } else {
    std::string guid =
        get_guid_with_precision("strided_slice_grad", grad_scalar_type);

    synSliceParamsV2 params;

    std::fill_n(params.axes, HABANA_DIM_MAX, 0);
    std::fill_n(params.starts, HABANA_DIM_MAX, 0);
    std::fill_n(params.ends, HABANA_DIM_MAX, 0);
    std::fill_n(params.steps, HABANA_DIM_MAX, 1);

    for (size_t i = 0; i < input_sizes.size(); ++i) {
      params.axes[i] = input_sizes.size() - i - 1;
      if (static_cast<long>(i) == dim) {
        params.starts[i] = normalize_idx(start, input_sizes[i]);
        params.ends[i] = normalize_idx(end, input_sizes[i]);
        params.steps[i] = step;
      } else {
        params.starts[i] = 0;
        params.ends[i] = input_sizes[i];
        params.steps[i] = 1;
      }
    }

    auto output = OpBackend::BuildNode(
        this,
        graph,
        {guid,
         {grad.syn_t},
         {{input_sizes, grad_scalar_type, 0}},
         &params,
         sizeof(params)});

    syn_out(0) = std::move(output[0]);
  }
}

} // namespace habana
