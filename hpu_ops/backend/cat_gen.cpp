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

namespace sh = synapse_helpers;

namespace habana {

void ValidateInputParams(
    const at::Stack& stack,
    std::vector<int64_t>& cal_out_shape) {
  auto tensors = stack[0].toTensorList().vec();
  auto out_result = stack[2].toTensor();
  auto in_tensor_count = tensors.size();
  auto input_tensor_type = tensors[0].scalar_type();
  auto input_tensor_dim = tensors[0].dim();

  for (unsigned i = 1; i < in_tensor_count; i++) {
    TORCH_CHECK(
        (input_tensor_dim == tensors[i].dim()),
        "Input tensor expected to be of same dimensions. Expected:",
        input_tensor_dim,
        ", got:",
        tensors[i].dim());
  }

  TORCH_CHECK(
      (out_result.dim() == cal_out_shape.size()),
      "Calculated Output tensor ambiguity with expected output dims. Calculated:",
      cal_out_shape.size(),
      ", got:",
      out_result.dim());

  TORCH_CHECK(
      (out_result.sizes() == cal_out_shape),
      "Calculated Output tensor shape is different than expected output shape. Calculated:",
      cal_out_shape,
      ", got:",
      out_result.sizes());
}

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

  std::vector<sh::tensor> cat_input_shTensor;
  std::vector<synTensor> cat_input_synTensor;

  auto in_tensors = stack[0].toTensorList().vec();
  auto out_tensor_type = in_tensors[0].scalar_type();

  for (unsigned i = 0; i < in_tensors.size(); i++) {
    if (in_tensors[i].scalar_type() != out_tensor_type) {
      cat_input_shTensor.emplace_back(CastHelper(
          graph,
          syn_in(i),
          in_tensors[i].sizes(),
          in_tensors[i].scalar_type(),
          out_tensor_type));
      cat_input_synTensor.emplace_back(cat_input_shTensor.back().get());
    } else {
      cat_input_synTensor.emplace_back(syn_in(i));
    }
  }

  auto cal_out_size = ComputeOutputShapes(stack)[0];
  ValidateInputParams(stack, cal_out_size);

  synConcatenateParams concat_params{};
  concat_params.axis = tensorlist[0].dim() - dim - 1;

  auto catop = BuildOp(
      graph,
      "concat",
      cat_input_synTensor,
      {{{cal_out_size}, result.scalar_type(), 0}},
      &concat_params,
      sizeof(concat_params));

  syn_out(0) = std::move(catop[0]);
}

} // namespace habana
