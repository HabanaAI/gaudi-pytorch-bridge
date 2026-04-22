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
#include "generated/backend/slice_backward.h"
#include "habana_helpers/conversion.h"

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

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = self.scalar_type();
  meta.shape = stack[1].toIntList().vec();

  return metaVec;
}

SharedMetaDataVector SliceBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
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
      SharedMetaDataVector meta;
      meta.reserve(1);
      auto& constantSharedMeta = meta.emplace_back("constant");
      constantSharedMeta.outputs_data.emplace_back(rank, dtype);
      return meta;
    }
    return {};
  } else {
    SharedMetaDataVector meta;
    meta.reserve(1);
    auto& stridedSliceGrad = meta.emplace_back("strided_slice_grad");
    stridedSliceGrad.inputs_data.emplace_back(rank, dtype);
    stridedSliceGrad.outputs_data.emplace_back(rank, dtype);

    return meta;
  }
}

void SliceBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "SliceBackward::AddNode");
  auto grad = stackGetter.getNextInput<TensorsPair>();
  auto input_sizes = stackGetter.getNextInput<std::vector<int64_t>>();
  auto dim = stackGetter.getNextInput<long>();
  auto start = stackGetter.getNextInput<long>();
  auto end = stackGetter.getNextInput<long>();
  auto step = stackGetter.getNextInput<long>();

  auto meta = SliceBackwardMeta(stack);

  const auto grad_sizes = meta[0].shape;
  const auto grad_scalar_type = meta[0].dtype;

  if (std::find(grad_sizes.begin(), grad_sizes.end(), 0) != grad_sizes.end()) {
    auto zero_tensor =
        ConstantHelper(graph, 0, grad_scalar_type, input_sizes, 0);
    syn_out(0) = std::move(zero_tensor);
  } else {
    using namespace std::literals;
    std::string guid =
        get_guid_with_precision("strided_slice_grad"sv, grad_scalar_type);

    synSliceParamsV2 params;

    std::fill_n(params.axes, HABANA_DIM_MAX, 0);
    std::fill_n(params.starts, HABANA_DIM_MAX, 0);
    std::fill_n(params.ends, HABANA_DIM_MAX, 0);
    std::fill_n(params.steps, HABANA_DIM_MAX, 1);

    for (size_t i = 0; i < input_sizes.size(); ++i) {
      params.axes[i] = safe_convert<unsigned int>(input_sizes.size() - i - 1);
      if (static_cast<long>(i) == dim) {
        params.starts[i] =
            static_cast<unsigned long>(normalize_idx(start, input_sizes[i]));
        params.ends[i] =
            static_cast<unsigned long>(normalize_idx(end, input_sizes[i]));
        params.steps[i] = static_cast<unsigned long>(step);
      } else {
        params.starts[i] = 0;
        params.ends[i] = static_cast<unsigned long>(input_sizes[i]);
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
