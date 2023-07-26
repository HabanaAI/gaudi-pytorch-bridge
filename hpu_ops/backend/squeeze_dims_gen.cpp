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

#include <algorithm>
#include "generated/backend/squeeze.h"

namespace sh = synapse_helpers;

namespace habana {

sizes_vec SqueezeDimsOutputShape(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dims = stack[1].toIntList().vec();
  auto output_shape = self.sizes().vec();

  if (output_shape.size() == 1) {
    return {output_shape};
  }

  at::wrap_all_dims(dims, self.dim());
  std::sort(dims.begin(), dims.end(), std::greater<int64_t>());

  for (auto dim : dims) {
    if (output_shape[dim] == 1) {
      output_shape.erase(output_shape.begin() + dim);
    }
  }

  return {output_shape};
}

void SqueezeDims::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "SqueezeDims::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto dims = getNextInput<std::vector<int64_t>>(stackGetter);
  auto rank = self.pt_t.dim();
  auto dtype = ScalarType();
  auto intermediate_shape = self.pt_t.sizes().vec();

  at::wrap_all_dims(dims, rank);
  std::vector<int64_t> valid_dims;
  for (auto dim : dims) {
    if (intermediate_shape[dim] == 1) {
      valid_dims.push_back(dim);
    }
  }

  if (valid_dims.empty() || rank == 1) {
    auto out = BuildOp(
        graph, "identity", {self.syn_t}, {{intermediate_shape, dtype, 0}});
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
