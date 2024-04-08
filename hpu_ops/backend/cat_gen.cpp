/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

#include "common/utils.h"
#include "generated/backend/cat.h"

namespace sh = synapse_helpers;

namespace habana {
OutputMetaDataVector CatMeta(const at::Stack& stack) {
  auto tensors_ = stack[0].toTensorVector();
  auto dim = stack[1].toInt();

  TORCH_CHECK(tensors_.size() > 0, "Empty tensors list!");
  const at::Tensor& first_tensor = tensors_[0];
  auto tensors = at::filter(tensors_, [](const at::Tensor& tensor) {
    return tensor.dim() != 1 || tensor.size(0) != 0;
  });

  std::vector<int64_t> ref_out_size;
  if (stack.size() > 2) {
    auto tensor_out = stack[2].toTensor();
    if (tensor_out.dim() != 1 || tensor_out.size(0) != 0)
      ref_out_size = tensor_out.sizes().vec();
  }

  std::vector<int64_t> out_size;
  if (tensors.size() > 0) {
    const at::Tensor& first_valid_tensor = tensors[0];
    dim = at::maybe_wrap_dim(dim, first_valid_tensor.dim());

    out_size = first_valid_tensor.sizes().vec();
    out_size[dim] = 0;
    for (const at::Tensor& tensor : tensors) {
      out_size[dim] += tensor.sizes()[dim];
    }
    if (!ref_out_size.empty()) {
      TORCH_CHECK(out_size[dim] == ref_out_size[dim], "Cat output mismatch");
    }
  }
  auto dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      {tensors_},
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  return {OutputMetaData{
      dtype,
      out_size,
      {},
      first_tensor.layout(),
      first_tensor.suggest_memory_format()}};
}

void CatHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto in_tensors = stack[0].toTensorList().vec();
  TORCH_CHECK(in_tensors.size() > 0, "Empty tensors list!");
  auto dim = stack[1].toInt();

  const auto md = OutputMeta(stack)[0];
  const auto cal_out_size = md.shape;
  const auto out_tensor_type = md.dtype;

  std::vector<size_t> valid_indices;
  valid_indices.reserve(in_tensors.size());

  for (size_t i{}; i < in_tensors.size(); ++i) {
    const auto& t = in_tensors[i];
    if (t.dim() != 1 || t.size(0) != 0) {
      valid_indices.push_back(i);
    }
  }

  if (valid_indices.empty()) {
    auto identity =
        BuildOp(graph, "memset", {}, {{cal_out_size, out_tensor_type, 0}});
    syn_out(0) = std::move(identity[0]);
    return;
  }

  int64_t first_valid_tensor_dim = in_tensors[valid_indices[0]].dim();
  dim = at::maybe_wrap_dim(dim, first_valid_tensor_dim);

  std::vector<sh::tensor> cat_input_shTensor;
  std::vector<synTensor> cat_input_synTensor;

  for (size_t i : valid_indices) {
    if (habana_helpers::pytorch_to_synapse_type(in_tensors[i].scalar_type()) !=
        habana_helpers::pytorch_to_synapse_type(out_tensor_type)) {
      cat_input_shTensor.emplace_back(BuildCast(
          this,
          graph,
          syn_in(i),
          in_tensors[i].sizes(),
          in_tensors[i].scalar_type(),
          out_tensor_type));
      cat_input_synTensor.push_back(cat_input_shTensor.back().get());
    } else {
      cat_input_synTensor.push_back(syn_in(i));
    }
  }

  synConcatenateParams concat_params{};
  concat_params.axis = first_valid_tensor_dim - dim - 1;

  CreateShapeTensorInput(
      graph, out_tensor_type, cal_out_size, cat_input_synTensor);
  auto catop = BuildOp(
      graph,
      "concat",
      std::move(cat_input_synTensor),
      {{{cal_out_size}, out_tensor_type, 0}},
      &concat_params,
      sizeof(concat_params));
  syn_out(0) = std::move(catop[0]);
}

} // namespace habana
