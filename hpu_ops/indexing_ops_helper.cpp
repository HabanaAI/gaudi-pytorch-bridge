/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/indexing_ops_helper.h"
#include <c10/core/ScalarType.h>
#include <stdint.h>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "hpu_ops/hpu_op_helper.h"
namespace habana {
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
    std::vector<int64_t> input_shape,
    std::vector<bool> adv_index_dims,
    std::vector<std::vector<int64_t>> indexing_tensor_shapes) {
  unsigned max_elem_count = 0;
  std::vector<int64_t> largest_specified_index_t_size;
  for (size_t i = 0; i < adv_index_dims.size(); ++i) {
    unsigned elem_count;
    if (i < indexing_tensor_shapes.size()) {
      elem_count = std::accumulate(
          indexing_tensor_shapes[i].begin(),
          indexing_tensor_shapes[i].end(),
          1,
          std::multiplies<unsigned>());
    } else {
      elem_count = input_shape[i];
    }
    if (!adv_index_dims[i] && elem_count > max_elem_count) {
      if (i < indexing_tensor_shapes.size())
        largest_specified_index_t_size = indexing_tensor_shapes[i];
      max_elem_count = elem_count;
    }
  }

  std::vector<int64_t> output_shape;
  bool non_adv_indexing_found = false;
  for (size_t i = 0; i < adv_index_dims.size(); ++i) {
    if (adv_index_dims[i]) {
      output_shape.emplace_back(
          (i < indexing_tensor_shapes.size()) ? indexing_tensor_shapes[i][0]
                                              : input_shape[i]);
    } else {
      if (!non_adv_indexing_found) {
        output_shape.insert(
            output_shape.cend(),
            largest_specified_index_t_size.cbegin(),
            largest_specified_index_t_size.cend());
        non_adv_indexing_found = true;
      }
    }
  }
  for (int64_t i = adv_index_dims.size();
       i < static_cast<int64_t>(input_shape.size());
       i++) {
    output_shape.emplace_back(input_shape[i]);
  }

  return output_shape;
}

bool hasContiguousSubspace(c10::ArrayRef<c10::IValue> indices_ival) {
  bool explicit_indices_together = false;
  int index_tensor_groups = 0;
  for (auto input : indices_ival) {
    auto o1 = input.toOptional<at::Tensor>();
    if (o1.has_value() && o1.value().defined()) {
      if (!explicit_indices_together) {
        explicit_indices_together = true;
        index_tensor_groups++;
      }
    } else {
      if (explicit_indices_together)
        explicit_indices_together = false;
    }
  }

  return index_tensor_groups <= 1;
}

int hasContiguousSubspace(std::vector<int64_t> implicit_indices_pos_vec) {
  bool explicit_indices_together = false;
  int index_tensor_groups = 0;
  int index_tensor_group_start = 0;
  int dim = 0;
  for (auto pos : implicit_indices_pos_vec) {
    if (pos == -1) {
      if (explicit_indices_together) {
        explicit_indices_together = false;
      }
    } else {
      if (!explicit_indices_together) {
        index_tensor_group_start = dim;
        index_tensor_groups++;
      }
      explicit_indices_together = true;
    }
    dim++;
  }
  if (index_tensor_groups <= 1)
    return index_tensor_group_start;
  else
    return 0;
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
// transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
std::tuple<std::vector<int64_t>, std::vector<at::Tensor>> transposeToFront(
    const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();
  std::vector<int64_t> dims;
  std::vector<at::Tensor> transposedIndices;
  std::vector<c10::optional<at::Tensor>> indices;
  for (const auto& index_opt : indices_ival) {
    auto o1 = index_opt.toOptional<at::Tensor>();
    if (!o1.has_value() || !o1.value().defined()) {
      indices.emplace_back(c10::nullopt);
    } else if (o1.has_value() && o1.value().defined()) {
      const auto& index = o1.value();
      indices.emplace_back(std::move(index));
    }
  }
  dims.reserve(self.dim());
  for (const auto i : c10::irange<size_t>(self.dim())) {
    if ((i < indices.size()) && indices[i].has_value()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i].value());
    }
  }
  for (const auto i : c10::irange<size_t>(self.dim())) {
    if ((i < indices.size()) && !indices[i].has_value()) {
      dims.push_back(i);
      // Don't add undefined tensors to list as Lazy infra can't handle such
      // tensors
    } else if ((i >= indices.size())) {
      dims.push_back(i);
    }
  }
  return std::make_tuple(dims, std::move(transposedIndices));
}

bool check_for_adv_indexing(c10::ArrayRef<c10::IValue> indices_in_orig) {
  bool advanced_indexing = false;
  c10::ScalarType prev_scalar_type{c10::ScalarType::Undefined};
  bool first_scalar = true;
  if (indices_in_orig.size() <= MAX_DIMS_FOR_ADVANCED_INDEXING) {
    for (auto input : indices_in_orig) {
      auto o1 = input.toOptional<at::Tensor>();
      if (!o1.has_value() || !o1->defined()) {
        advanced_indexing = true;
        break;
      } else if (o1.has_value() && o1->defined()) {
        // if we are indexing using a mixture of long and boolean indices,then
        // also we will work in advanced indexing mode
        if (first_scalar) {
          first_scalar = false;
          prev_scalar_type = o1.value().scalar_type();
        }
        if (prev_scalar_type != o1.value().scalar_type()) {
          advanced_indexing = true;
          break;
        }
        prev_scalar_type = o1.value().scalar_type();
      }
    }
  }
  return advanced_indexing;
}

bool handle_bool_mask_indices(
    c10::ArrayRef<c10::IValue>& indices_in_orig,
    std::vector<c10::IValue>& indices_in_ivals_vec,
    std::vector<c10::optional<at::Tensor>>& bool_indices_vec) {
  at::Tensor t_nz;
  bool has_bool_mask = false;

  for (auto input : indices_in_orig) {
    auto o1 = input.toOptional<at::Tensor>();
    if (o1.has_value() && !o1->defined()) {
      bool_indices_vec.emplace_back(o1.value());
      indices_in_ivals_vec.push_back(c10::IValue(o1.value()));
    } else if (o1.has_value() && o1->defined()) {
      if (o1.value().scalar_type() == c10::ScalarType::Bool) {
        if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
          has_bool_mask = true;
          auto nonzero_indices = habana_lazy::nonzero_hpu_lazy(o1.value());
          t_nz = habana_lazy::squeeze_hpu_lazy(nonzero_indices, 1);
          if (t_nz.dim() > 1) {
            std::vector<int64_t> dims_sz_vec(t_nz.sizes()[1], 1);
            c10::IntArrayRef dims_sz(dims_sz_vec);
            auto nz_indices =
                habana_lazy::split_with_sizes_hpu_lazy(t_nz, dims_sz, 1);
            for (auto i : c10::irange((int)nz_indices.size())) {
              auto nzi = habana_lazy::squeeze_hpu_lazy(nz_indices.at(i), 1);
              bool_indices_vec.emplace_back(nzi);
              indices_in_ivals_vec.emplace_back(c10::IValue(nzi));
            }
          } else {
            bool_indices_vec.emplace_back(t_nz);
            indices_in_ivals_vec.emplace_back(c10::IValue(t_nz));
          }
        } else {
          has_bool_mask = true;
          auto nonzero_indices = at::nonzero(o1.value());
          t_nz = at::squeeze(nonzero_indices, 1);
          if (t_nz.dim() > 1) {
            std::vector<int64_t> dims_sz_vec(t_nz.sizes()[1], 1);
            c10::IntArrayRef dims_sz(dims_sz_vec);
            auto nz_indices = at::split_with_sizes(t_nz, dims_sz, 1);
            for (auto i : c10::irange((int)nz_indices.size())) {
              auto nzi = at::squeeze(nz_indices.at(i), 1).contiguous();
              bool_indices_vec.emplace_back(nzi);
              indices_in_ivals_vec.emplace_back(c10::IValue(nzi));
            }
          } else {
            bool_indices_vec.emplace_back(t_nz);
            indices_in_ivals_vec.emplace_back(c10::IValue(t_nz));
          }
        }
      } else {
        bool_indices_vec.emplace_back(o1.value());
        indices_in_ivals_vec.push_back(c10::IValue(o1.value()));
      }
    }
  }
  return has_bool_mask;
}

std::vector<std::vector<int64_t>> calc_indexing_tensors_shapes(
    const at::Stack& stack) {
  std::vector<std::vector<int64_t>> indexing_sizes;
  at::Tensor self = stack_tensor(stack, 0);
  auto self_size = self.sizes().vec();
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();
  std::vector<bool> adv_ind_dim = stack[2].toBoolList().vec();
  const bool adv_indexing_present =
      std::any_of(adv_ind_dim.cbegin(), adv_ind_dim.cend(), [](const auto& i) {
        return i == true;
      });
  if (adv_indexing_present) {
    const std::vector<int64_t> self_permute_dims = stack[3].toIntList().vec();
    std::vector<int64_t> permuted_self_size(self_size.size(), 0);
    for (const auto i : c10::irange(self_size.size())) {
      permuted_self_size[i] = self_size[self_permute_dims[i]];
    }

    int idim = 0;
    for (const auto i : c10::irange(adv_ind_dim.size())) {
      if (adv_ind_dim[i]) {
        indexing_sizes.emplace_back(
            std::vector<int64_t>{permuted_self_size[i]});
      } else {
        indexing_sizes.emplace_back(
            indices_ival[idim].toTensor().sizes().vec());
        idim++;
      }
    }
  } else {
    for (const auto i : c10::irange(adv_ind_dim.size())) {
      indexing_sizes.emplace_back(indices_ival[i].toTensor().sizes().vec());
    }
  }
  return indexing_sizes;
}

std::tuple<bool, int, std::vector<int64_t>, std::vector<at::Tensor>>
generate_advanced_indexing_indices_list(const at::Stack& stack) {
  at::Tensor self = stack_tensor(stack, 0);
  auto self_size = self.sizes().vec();
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();

  std::vector<at::Tensor> indices;
  std::vector<int64_t> self_permute_dims(self.dim());
  std::vector<int64_t> implicit_indices_pos_vec(indices_ival.size());
  bool dims_permuted = false;
  int num_index_tensors = 0;
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  auto isSpaceContiguous = hasContiguousSubspace(indices_ival);
  if (!isSpaceContiguous) {
    std::tie(self_permute_dims, indices) = transposeToFront(stack);
    dims_permuted = true;
    for (int i = 0; i < (int)indices.size(); i++) {
      implicit_indices_pos_vec[i] = indices[i].sizes()[0];
      num_index_tensors++;
    }
    for (int i = (int)indices.size(); i < (int)indices_ival.size(); i++) {
      implicit_indices_pos_vec[i] = -1; //-1 indicates implicit indexing
      num_index_tensors++;
    }
  } else {
    for (const auto i : c10::irange(self.dim())) {
      self_permute_dims[i] = i;
    }
    int i = 0;
    for (const auto& index_opt : indices_ival) {
      auto o1 = index_opt.toOptional<at::Tensor>();
      if (!o1.has_value() || !o1.value().defined()) {
        // Don't add undefined tensors to list as Lazy infra can't handle such
        // tensors
        implicit_indices_pos_vec[i] = -1; //-1 indicates implicit indexing
      } else if (o1.has_value() && o1.value().defined()) {
        const auto& index = o1.value();
        indices.emplace_back(std::move(index));
        implicit_indices_pos_vec[i] = index.sizes()[0];
      }
      i++;
    }
    num_index_tensors = (int)indices_ival.size();
  }

  auto broadcast_to_this_size = broadcast_size(indices);
  for (auto& tensor : indices)
    tensor = at::broadcast_to(tensor, broadcast_to_this_size);

  //"self" is not yet permuted for advanced indexing, but it has to be
  // considered permuted while using self's sizes in computations
  return std::make_tuple(
      dims_permuted, num_index_tensors, self_permute_dims, indices);
}

// index is implemented using mxnet_gatherNd, refer below for output shape
// computation
// ref:https://github.com/apache/incubator-mxnet/blob/master/src/operator/tensor/indexing_op.h#L1319
std::vector<int64_t> ComputeIndexOperatorOutputShape(
    const at::Tensor& input,
    at::TensorList indices) {
  auto input_shape = input.sizes();
  auto indices_shape = indices_size(indices);

  if (input.dim() == 0 && input.numel() == 1)
    return {input.sizes().vec()};

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

std::vector<int64_t> get_index_result_shape(
    std::vector<at::IValue> inputs_vec) {
  const at::Tensor input = inputs_vec[0].toTensor();
  auto indices = inputs_vec[1].toTensorList().vec();
  auto adv_index_dims = inputs_vec[2].toBoolList().vec();
  std::vector<int64_t> implicit_indices_pos_vec =
      inputs_vec[3].toIntList().vec();
  bool adv_indexing_present = false;
  for (auto i : implicit_indices_pos_vec) {
    if (i == -1) {
      adv_indexing_present = true;
      break;
    }
  }
  if (adv_indexing_present) {
    std::vector<int64_t> self_permute_dims = inputs_vec[4].toIntList().vec();
    std::vector<int64_t> permuted_input_sizes, new_strides;
    std::tie(permuted_input_sizes, new_strides) =
        PermuteOperator::compute_output_shape(input, self_permute_dims);

    auto indexing_tensor_shapes = calc_indexing_tensors_shapes(inputs_vec);
    auto shape = habana::ComputeOutputShapeWithAdvIndexing(
        permuted_input_sizes, adv_index_dims, indexing_tensor_shapes);
    // TO DO: Resize output tensor if not of required shape.
    return shape;
  } else {
    auto shape = ComputeIndexOperatorOutputShape(input, indices);
    // TO DO: Resize output tensor if not of required shape.
    return shape;
  }
}

} // namespace habana
