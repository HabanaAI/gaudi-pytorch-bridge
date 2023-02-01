/******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/cat.h"

namespace habana {

sizes_vec CatOutOutputShape(const at::Stack& stack) {
  auto tensors = stack[0].toTensorList().vec();
  auto dim_ = stack[1].toInt();
  int64_t dim = at::maybe_wrap_dim(
      dim_,
      tensors[0].dim(),
      /*wrap_scalar=*/true);

  auto in_tensor_count = tensors.size();
  auto first_tensor = tensors[0];
  auto out_size = first_tensor.sizes().vec();
  out_size[dim] = 0;
  for (unsigned i = 0; i < in_tensor_count; i++) {
    out_size[dim] += tensors[i].sizes()[dim];
  }
  return {out_size};
}

void CatOutHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto tensorlist = stack[0].toTensorList().vec();
  auto dim_ = stack[1].toInt();
  auto result = stack[2].toTensor();
  TORCH_CHECK(tensorlist.size() > 0, "Empty tensors list!");
  int64_t dim = at::maybe_wrap_dim(
      dim_,
      tensorlist[0].dim(),
      /*wrap_scalar=*/true);

  std::vector<synTensor> cat_input_synTensor;

  for (int i = 0; i < (int)tensorlist.size(); i++) {
    cat_input_synTensor.emplace_back(syn_in(i));
  }

  auto out_size = ComputeOutputShapes(stack)[0];

  synConcatenateParams concat_params{};
  concat_params.axis = tensorlist[0].dim() - dim - 1;

  auto catop = BuildOp(
      graph,
      "concat",
      cat_input_synTensor,
      {{{out_size}, result.scalar_type(), 0}},
      &concat_params,
      sizeof(concat_params));

  syn_out(0) = std::move(catop[0]);
}

} // namespace habana
