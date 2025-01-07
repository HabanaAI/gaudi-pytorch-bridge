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

SharedMetaDataVector SliceBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto dtype = grad.scalar_type();
  const auto inputSizes = stack.at(1).toIntList();
  const auto rank = inputSizes.size();

  if (std::find(std::begin(inputSizes), std::end(inputSizes), 0) !=
      std::end(inputSizes)) {
    if (std::accumulate(
            std::begin(inputSizes),
            std::end(inputSizes),
            1,
            std::multiplies<int>()) > 1) {
      SharedMetaData constantSharedMeta{"constant"};
      constantSharedMeta.outputs_data.emplace_back(rank, dtype);
      return {constantSharedMeta};
    }
    return {};
  } else {
    SharedMetaData stridedSliceGrad{"strided_slice_grad"};
    stridedSliceGrad.inputs_data.emplace_back(rank, dtype);
    stridedSliceGrad.outputs_data.emplace_back(rank, dtype);

    return {stridedSliceGrad};
  }
}

void SliceBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "SliceBackward::AddNode");
  auto grad = stackGetter.getNextInput<TensorsPair>();
  auto input_sizes = stackGetter.getNextInput<std::vector<int64_t>>();
  auto dim = stackGetter.getNextInput<int>();
  auto start = stackGetter.getNextInput<int>();
  auto end = stackGetter.getNextInput<int>();
  auto step = stackGetter.getNextInput<int>();

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
