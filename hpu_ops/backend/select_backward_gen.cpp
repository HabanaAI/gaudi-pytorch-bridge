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

SharedMetaDataVector SelectBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  auto dtype = grad.scalar_type();
  auto rank = stack.at(1).toIntList().size();

  SharedMetaData stridedSliceGrad{"strided_slice_grad"};
  stridedSliceGrad.inputs_data.emplace_back(rank, dtype);
  stridedSliceGrad.outputs_data.emplace_back(rank, dtype);
  return {stridedSliceGrad};
}

static int normalize_idx(int idx, int64_t size) {
  if (size <= 0) {
    return 0;
  }

  if (idx < 0) {
    idx += size;
    if (idx < 0) {
      idx %= size;
      if (idx < 0) {
        idx += size;
      }
    }
  } else if (idx >= size) {
    idx -= size;
    if (idx >= size) {
      idx %= size;
    };
  }

  return idx;
}

void SelectBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "SelectBackward::AddNode");
  auto grad = stackGetter.getNextInput<TensorsPair>();
  auto input_sizes = stackGetter.getNextInput<std::vector<int64_t>>();
  auto dim = stackGetter.getNextInput<int>();
  auto index = stackGetter.getNextInput<int>();

  dim = at::maybe_wrap_dim(dim, input_sizes.size());

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

    auto dim_tpc = get_dim_in_tpc_order(dim, grad_shape.size() + 1);

    reshaped_grad = OpBackend::BuildExpandDims(
        this, graph, grad.syn_t, reshaped_grad_size, grad_scalar_type, dim_tpc);
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
