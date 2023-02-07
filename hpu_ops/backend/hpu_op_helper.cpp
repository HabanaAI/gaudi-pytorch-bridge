/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    metatensors.emplace_back(at::empty(
        tensor.sizes(),
        tensor.options().device(at::kMeta),
        tensor.suggest_memory_format()));
  }
  return metatensors;
}

std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (tensor.has_value()) {
      const auto& tv = tensor.value();
      metatensors.emplace_back(at::empty(
          tv.sizes(),
          tv.options().device(at::kMeta),
          tv.suggest_memory_format()));
    } else {
      metatensors.emplace_back(tensor);
    }
  }
  return metatensors;
}

// brodcast index tensor shape and get the correct shape and size
static std::vector<int64_t> broadcast_size(at::TensorList indices) {
  auto size = indices[0].sizes().vec();
  for (size_t i = 1; i < indices.size(); i++) {
    size = at::infer_size(size, indices[i].sizes());
  }
  return size;
}

// get the first index tensor shape and size
std::vector<int64_t> indices_size(at::TensorList indices) {
  auto first_size = broadcast_size(indices);

  int64_t in_tensor_count = indices.size(); // num input tensors

  std::vector<int64_t> out_size{in_tensor_count};
  out_size.insert(out_size.end(), first_size.begin(), first_size.end());

  return out_size;
}

// index is implemented using mxnet_gatherNd, refer below for output shape
// computation
// ref:https://github.com/apache/incubator-mxnet/blob/master/src/operator/tensor/indexing_op.h#L1319
std::vector<int64_t> ComputeOutputShapeWithAdvIndexing(
    const at::Tensor& input,
    at::TensorList indices,
    c10::List<int64_t> adv_index_dims,
    bool get_adv_indexing_out_shape) {
  auto input_shape = input.sizes();
  auto indices_shape = indices_size(indices);
  bool advanced_indexing = false;
  if (input.dim() == 0 && input.numel() == 1)
    return {input.sizes().vec()};

  for (int i = 0; i < adv_index_dims.size(); i++) {
    if (adv_index_dims[i]) {
      advanced_indexing = true;
      break;
    }
  }
  if (get_adv_indexing_out_shape && advanced_indexing) {
    std::vector<int64_t> output_shape;
    int64_t largest_specified_index_t_size = 0;
    for (int i = 0; i < input.dim(); i++) {
      if (adv_index_dims[i] > largest_specified_index_t_size)
        largest_specified_index_t_size = adv_index_dims[i];
    }
    bool explicit_index_found = false;
    for (int i = 0; i < input.dim(); i++) {
      if (adv_index_dims[i]) { // dim has explicit index tensor
        if (!explicit_index_found) {
          output_shape.emplace_back(largest_specified_index_t_size);
          explicit_index_found = true;
        }
      } else { // advanced indexing done on this dim
        output_shape.emplace_back(input_shape[i]);
      }
    }
    return output_shape;
  } else {
    auto output_rank = static_cast<int64_t>(
        indices_shape.size() + input.ndimension() - indices_shape[0] - 1);
    std::vector<int64_t> output_shape(output_rank, -1);
    for (size_t i = 0; i < indices_shape.size() - 1; i++) {
      output_shape[i] = indices_shape[i + 1];
    }
    for (int64_t i = 0;
         i < static_cast<int64_t>(input.ndimension() - indices_shape[0]);
         i++) {
      output_shape[indices_shape.size() - 1 + i] =
          input_shape[indices_shape[0] + i];
    }
    return output_shape;
  }
}

} // namespace habana
