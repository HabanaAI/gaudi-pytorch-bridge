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
#include "generated/backend/select_backward.h"

using namespace synapse_helpers::layouts;

namespace habana {

OutputMetaDataVector SelectBackwardMeta(const at::Stack& stack) {
  auto self = stack[0].toTensor();
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = stack[1].toIntList().vec();

  return {meta};
}

template <typename idx_t, typename size_t>
idx_t normalize_idx(idx_t idx, size_t size) {
  if (size <= 0) {
    return 0;
  }

  if (idx < 0) {
    do {
      idx += size;
    } while (idx < 0);
  } else {
    while (idx > size) {
      idx -= size;
    };
  }

  return idx;
}

void SelectBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "SelectBackward::AddNode");
  auto grad = getNextInput<TensorsPair>(stackGetter);
  auto input_sizes = getNextInput<std::vector<int64_t>>(stackGetter);
  auto dim = getNextInput<int>(stackGetter);
  auto index = getNextInput<int>(stackGetter);

  const auto grad_scalar_type = grad.pt_t.scalar_type();
  const auto grad_shape = grad.pt_t.sizes();

  std::string guid =
      get_guid_with_precision("strided_slice_grad", grad_scalar_type);

  synSliceParamsNDims params;

  std::fill_n(params.axes, HABANA_DIM_MAX, 0);
  std::fill_n(params.starts, HABANA_DIM_MAX, 0);
  std::fill_n(params.ends, HABANA_DIM_MAX, 0);
  std::fill_n(params.steps, HABANA_DIM_MAX, 1);

  // Synapse indexes dims in opposite order then PT
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    params.axes[i] = input_sizes.size() - i - 1;
    if (static_cast<int>(i) == dim) {
      params.starts[i] = normalize_idx(index, input_sizes[i]);
      params.ends[i] = params.starts[i] + 1;
    } else {
      params.starts[i] = 0;
      params.ends[i] = input_sizes[i];
    }
  }

  // strided_slice_grad requires sliced tensor (grad) to have the same
  // dimensions as unsliced tensor (defined by input_sizes)
  bool is_reshape_required =
      grad.pt_t.dim() > 0 && !(grad.pt_t.dim() == 1 && grad_shape.at(0) == 1);

  std::optional<synapse_helpers::tensor> reshaped_grad;

  if (is_reshape_required) {
    std::vector<int64_t> reshaped_grad_size = grad_shape.vec();

    reshaped_grad_size.insert(reshaped_grad_size.begin() + dim, 1);

    reshaped_grad = OpBackend::BuildReshape(
        this, graph, grad.syn_t, reshaped_grad_size, grad_scalar_type);
  }

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {is_reshape_required ? (*reshaped_grad).get() : grad.syn_t},
       {{input_sizes, grad_scalar_type, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

} // namespace habana
