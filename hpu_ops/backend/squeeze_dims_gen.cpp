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

#include <algorithm>
#include "generated/backend/squeeze.h"

namespace sh = synapse_helpers;

namespace habana {

OutputMetaDataVector SqueezeDimsMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dims = stack[1].toIntList().vec();
  auto output_shape = self.sizes().vec();
  OutputMetaData meta;
  meta.dtype = self.scalar_type();

  if (output_shape.size() == 1 && dims.size() == 1 && output_shape[0] == 1) {
    meta.shape = {};
  } else if (output_shape.size() == 1 || output_shape.size() == 0) {
    meta.shape = output_shape;
  } else {
    at::wrap_all_dims(dims, self.dim());
    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());

    for (auto dim : dims) {
      if (output_shape[dim] == 1) {
        output_shape.erase(output_shape.begin() + dim);
      }
    }
    meta.shape = output_shape;
  }

  return {meta};
}

void SqueezeDims::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "SqueezeDims::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto dims = stackGetter.getNextInput<std::vector<int64_t>>();
  auto rank = self.pt_t.dim();
  auto meta = SqueezeDimsMeta(stack)[0];
  auto output_shape = meta.shape;
  auto dtype = meta.dtype;
  auto intermediate_shape = self.pt_t.sizes().vec();

  at::wrap_all_dims(dims, rank);
  std::vector<int64_t> valid_dims;
  for (auto dim : dims) {
    if (intermediate_shape.size() > static_cast<uint64_t>(dim) &&
        intermediate_shape[dim] == 1) {
      valid_dims.push_back(dim);
    }
  }

  if (valid_dims.empty() || rank == 1 || rank == 0) {
    auto out =
        BuildOp(graph, "identity", {self.syn_t}, {{output_shape, dtype, 0}});
    syn_out(0) = std::move(out[0]);
    return;
  }

  std::sort(valid_dims.begin(), valid_dims.end(), std::greater<int64_t>());

  std::vector<sh::tensor> intermediate_syn_helpers;
  std::vector<synTensor> intermediate_syn_tensors{self.syn_t};

  c10::optional<int> result_idx = c10::nullopt;
  auto dims_count = valid_dims.size();

  for (size_t i = 0; i < dims_count; ++i) {
    auto dim = valid_dims[i];
    intermediate_shape.erase(intermediate_shape.begin() + dim);
    const auto syn_axis = (rank--) - dim - 1;
    synAxisParams params{static_cast<unsigned int>(syn_axis)};

    if (i == dims_count - 1) {
      result_idx = c10::make_optional<int>(0);
    }

    intermediate_syn_helpers.emplace_back(std::move(OpBackend::BuildNode(
        this,
        graph,
        {"squeeze",
         {intermediate_syn_tensors.back()},
         {{intermediate_shape, dtype, result_idx}},
         &params,
         sizeof(params)})[0]));
    intermediate_syn_tensors.emplace_back(
        intermediate_syn_helpers.back().get());
  }

  syn_out(0) = std::move(intermediate_syn_helpers.back());
}

} // namespace habana
