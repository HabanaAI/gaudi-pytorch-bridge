/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include "generated/lazy/gather.h"
#include "generated/lazy/index.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/tensor_shape_kernels.h"

#define MAX_DIMS_FOR_ADVANCED_INDEXING (8)

namespace habana {

FALLBACK_CHECK(
    IndexPutFallbackCheck,
    const c10::List<c10::optional<at::Tensor>>& indices) {
  at::Stack stack = {indices};
  c10::ArrayRef<c10::IValue> indices_in = stack.at(0).toListRef();
  for (auto input : indices_in) {
    auto o1 = input.toOptional<at::Tensor>();
    if (o1.has_value() && o1.value().defined()) {
      continue;
    } else {
      return false; // advanced indexing is currently unsupported on HPU -
                    // fallback to CPU
    }
  }
  return true;
};

FALLBACK_CHECK(
    IndexFallbackCheck,
    [[maybe_unused]] const c10::List<c10::optional<at::Tensor>>& indices) {
  return true;
};

static C10_UNUSED std::vector<at::Tensor> expandTensors(
    const at::Tensor& self,
    at::IOptTensorListRef indices) {
  // If indices come in as ByteTensor or BoolTensor (masks), expand them into
  // the equivalent indexing by LongTensors
  std::vector<at::Tensor> result;
  for (const auto& index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back();
    } else {
      const auto& index = *index_opt;
      if (index.scalar_type() == c10::kByte ||
          index.scalar_type() == c10::kBool) {
        if (index.scalar_type() == c10::kByte) {
          TORCH_WARN(
              "indexing with dtype torch.uint8 is now deprecated,"
              " please use a dtype torch.bool instead.");
        }
        // The sizes of the ByteTensor mask or bool tensor must match the sizes
        // of the corresponding dimensions in self
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = result.size() + j;
          if (index.size(j) != self.size(srcIdx)) {
            // invalid_mask(self, srcIdx, index, j);
            TORCH_CHECK(
                0,
                "Indexing: The sizes of the ByteTensor mask or bool tensor must match the sizes of the corresponding dimensions in self");
          }
        }
        // Replace with nonzeros
        auto nonzero = index.nonzero();
        for (const auto j : c10::irange(index.dim())) {
          result.emplace_back(nonzero.select(1, j));
        }
      } else {
        result.emplace_back(std::move(index));
      }
    }
  }
  return result;
}

static C10_UNUSED int hasContiguousSubspace(
    c10::ArrayRef<c10::IValue> indices_ival) {
  bool explicit_indices_together = false;
  int index_tensor_groups = 0;
  int index_tensor_group_start = 0;
  int dim = 0;
  for (auto input : indices_ival) {
    auto o1 = input.toOptional<at::Tensor>();
    if (o1.has_value() && !o1->defined()) {
      if (explicit_indices_together) {
        explicit_indices_together = false;
      }
    } else if (o1.has_value() && o1->defined()) {
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
static C10_UNUSED std::tuple<std::vector<int64_t>, std::vector<at::Tensor>>
transposeToFront(const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();
  std::vector<int64_t> dims;
  std::vector<at::Tensor> transposedIndices;
  std::vector<c10::optional<at::Tensor>> indices;
  for (const auto& index_opt : indices_ival) {
    auto o1 = index_opt.toOptional<at::Tensor>();
    if (o1.has_value() && !o1.value().defined()) {
      indices.emplace_back(c10::nullopt);
    } else if (o1.has_value() && o1.value().defined()) {
      const auto& index = o1.value();
      indices.emplace_back(std::move(index));
    }
  }
  dims.reserve(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].has_value()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i].value());
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].has_value()) {
      dims.push_back(i);
      // Don't add undefined tensors to list as Lazy infra can't handle such
      // tensors
    }
  }
  return std::make_tuple(dims, std::move(transposedIndices));
}

static std::
    tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<at::Tensor>>
    generate_advanced_indexing_indices_list(const at::Stack& stack) {
  at::Tensor self = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();

  std::vector<at::Tensor> indices;
  std::vector<int64_t> self_permute_dims(self.dim());
  std::vector<int64_t> implicit_indices_pos_vec(indices_ival.size());
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  auto explicit_index_tensor_group_start = hasContiguousSubspace(indices_ival);
  if (!explicit_index_tensor_group_start) {
    std::tie(self_permute_dims, indices) = transposeToFront(stack);
    for (int i = 0; i < (int)indices.size(); i++) {
      implicit_indices_pos_vec[i] = i;
    }
    for (int i = (int)indices.size(); i < (int)indices_ival.size(); i++) {
      implicit_indices_pos_vec[i] = -1; //-1 indicates implicit indexing
    }
  } else {
    for (const auto i : c10::irange(self.dim())) {
      self_permute_dims[i] = i;
    }
    int i = 0;
    for (const auto& index_opt : indices_ival) {
      auto o1 = index_opt.toOptional<at::Tensor>();
      if (o1.has_value() && !o1.value().defined()) {
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
  }
  //"self" is not yet permuted for advanced indexing, but it has to be
  // considered permuted while using self's sizes in computations
  return std::make_tuple(implicit_indices_pos_vec, self_permute_dims, indices);
}

template <>
LazyIndex<at::Tensor>::LazyIndex(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs_orig,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(
          qualstring,
          inputs_orig,
          out_shapes_fn,
          -1) {
  habana_lazy::NoAccThread no_acc_thread;
  auto& sub_inputs = get_inputs();
  const at::Tensor self = sub_inputs.at(0).toTensor();
  c10::ArrayRef<c10::IValue> indices_in_orig = sub_inputs.at(1).toListRef();
  std::vector<at::IValue> inputs = inputs_orig;
  c10::ArrayRef<c10::IValue> indices_in;
  std::vector<c10::IValue> indices_in_ivals_vec;
  std::vector<c10::optional<at::Tensor>> bool_indices_vec;
  std::vector<at::Tensor> indices_vec_out{};
  std::vector<at::Tensor> indices_vec;
  TORCH_CHECK(
      self.dim() <= MAX_DIMS_FOR_ADVANCED_INDEXING,
      "Index op doesn't support more than ",
      MAX_DIMS_FOR_ADVANCED_INDEXING,
      " dims");
  bool advanced_indexing = false;
  std::array<int64_t, MAX_DIMS_FOR_ADVANCED_INDEXING> advanced_indexing_dims = {
      -1, -1, -1, -1, -1, -1, -1, -1};
  int dim = 0;
  int num_explicit_indices = 0;
  int broadcast_to_size = 0;
  bool explicit_indices_together = false;
  int index_tensor_groups = 0;
  int index_tensor_group_start = 0;
  int index_tensor_group_end = (int)indices_in.size();
  at::Tensor t_nz;
  bool has_bool_mask = false;
  c10::ScalarType prev_scalar_type;
  bool first_scalar = true;

  if (indices_in.size() <= MAX_DIMS_FOR_ADVANCED_INDEXING) {
    for (auto input : indices_in_orig) {
      auto o1 = input.toOptional<at::Tensor>();
      if (o1.has_value() && !o1->defined()) {
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
  if (advanced_indexing) {
    for (auto input : indices_in_orig) {
      auto o1 = input.toOptional<at::Tensor>();
      if (o1.has_value() && !o1->defined()) {
        bool_indices_vec.emplace_back(o1.value());
        indices_in_ivals_vec.push_back(c10::IValue(o1.value()));
      } else if (o1.has_value() && o1->defined()) {
        if (o1.value().scalar_type() == c10::ScalarType::Bool) {
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
          bool_indices_vec.emplace_back(o1.value());
          indices_in_ivals_vec.push_back(c10::IValue(o1.value()));
        }
      }
    }
  }
  if (advanced_indexing && has_bool_mask) {
    indices_in = indices_in_ivals_vec;
    c10::List<c10::optional<at::Tensor>> bool_mask_indices(bool_indices_vec);
    inputs.clear();
    inputs.emplace_back(sub_inputs.at(0));
    inputs.emplace_back(c10::IValue(bool_mask_indices));
  } else {
    indices_in = indices_in_orig;
  }

  at::Tensor self_permuted = self;
  std::vector<int64_t> implicit_indices_pos_vec;
  std::vector<int64_t> self_permute_dims;
  if (advanced_indexing) {
    if (indices_in.size() <= MAX_DIMS_FOR_ADVANCED_INDEXING) {
      for (auto input : indices_in) {
        auto o1 = input.toOptional<at::Tensor>();
        if (o1.has_value() && !o1->defined()) {
          advanced_indexing_dims[dim] =
              0; // to indicate to use self_sizes[dim] in shape calculations.
          if (explicit_indices_together) {
            explicit_indices_together = false;
            index_tensor_group_end = dim;
          }
        } else if (o1.has_value() && o1->defined()) {
          if (!explicit_indices_together) {
            index_tensor_group_start = dim;
            index_tensor_groups++;
          }
          explicit_indices_together = true;
          auto o1_sizes = o1.value().sizes().vec();
          advanced_indexing_dims[dim] = o1_sizes[0];
          if (advanced_indexing_dims[dim] > broadcast_to_size) {
            broadcast_to_size = advanced_indexing_dims[dim];
          }
          num_explicit_indices++;
        }
        if (explicit_indices_together) {
          index_tensor_group_end = dim;
        }
        dim++;
      }
    }
    if ((long)indices_in.size() < self.dim()) {
      for (int i = (int)indices_in.size(); i < self.dim(); i++) {
        advanced_indexing_dims[i] = 0; // to indicate to use self_sizes[i];
      }
    }
    std::tie(implicit_indices_pos_vec, self_permute_dims, indices_vec) =
        generate_advanced_indexing_indices_list(inputs); //(self, indices_in);
    if ((index_tensor_groups > 1) &&
        (num_explicit_indices >
         1)) { // all explicitly indexed dims are now in higher order dims.
      for (const auto i : c10::irange((int)implicit_indices_pos_vec.size())) {
        if (i < num_explicit_indices) {
          advanced_indexing_dims[i] = broadcast_to_size;
        } else {
          advanced_indexing_dims[i] = 0;
        }
      }
    } else if (num_explicit_indices > 1) {
      for (const auto i : c10::irange((int)implicit_indices_pos_vec.size())) {
        if ((i >= index_tensor_group_start) && (i <= index_tensor_group_end)) {
          advanced_indexing_dims[i] = broadcast_to_size;
        } else {
          advanced_indexing_dims[i] = 0;
        }
      }
    }
  } else {
    for (auto input : indices_in) {
      auto o1 = input.toOptional<at::Tensor>();
      if (o1.has_value() && o1->defined()) {
        indices_vec.push_back(o1.value());
      }
    }
  }
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }

  // handle views for tensorlist indices
  at::TensorList indices_in_list(indices_vec);
  indices_vec =
      habana_lazy::HbLazyTensorViews::HandleViewsTensorList(indices_in_list);
  // for case where indices are Boolean tensor(s), convert these to integer
  // indices using nonzero operator before calling index
  auto bool_non_adv_indexing_case =
      (!advanced_indexing &&
       (indices_vec[0].scalar_type() == c10::ScalarType::Bool));
  if (bool_non_adv_indexing_case) {
    for (size_t i = 0; i < indices_vec.size(); i++) {
      auto list = torch::nonzero_numpy(indices_vec.at(i));
      indices_vec_out.insert(
          indices_vec_out.cend(), list.cbegin(), list.cend());
    }
  }

  at::TensorList indices =
      (bool_non_adv_indexing_case) ? indices_vec_out : indices_vec;

  auto indices_out_vec =
      habana_lazy::HbLazyTensorViews::HandleViewsTensorList(indices);
  at::TensorList indices_out_list(indices_out_vec);
  get_inputs().at(0) = self_permuted;
  get_inputs().back() = indices_out_list;
  get_inputs().emplace_back(advanced_indexing_dims);
  get_inputs().emplace_back(implicit_indices_pos_vec);
  get_inputs().emplace_back(self_permute_dims);
}

template <>
at::Tensor LazyIndex<at::Tensor>::get_result_overrideable() {
  auto inputs = get_inputs();
  const at::Tensor input = inputs[0].toTensor();
  auto indices = inputs[1].toTensorList().vec();
  auto adv_index_dims = inputs[2].toIntList();
  std::vector<int64_t> implicit_indices_pos_vec = inputs[3].toIntList().vec();
  bool adv_indexing_present = false;
  for (auto i : implicit_indices_pos_vec) {
    if (i == -1) {
      adv_indexing_present = true;
      break;
    }
  }
  if (adv_indexing_present) {
    std::vector<int64_t> self_permute_dims = inputs[4].toIntList().vec();
    std::vector<int64_t> new_sizes, new_strides;
    std::tie(new_sizes, new_strides) =
        PermuteOperator::compute_output_shape(input, self_permute_dims);

    std::vector<int64_t> permuted_input_sizes;
    if (adv_indexing_present) {
      permuted_input_sizes = new_sizes;
    } else {
      permuted_input_sizes = input.sizes().vec();
    }

    auto shape = habana::ComputeOutputShapeWithAdvIndexing(
        permuted_input_sizes, indices, adv_index_dims, true);
    return habana_lazy::empty_hpu_lazy(
        shape, input.options(), input.suggest_memory_format(), false);
  } else {
    auto shape = IndexOperator::compute_output_shape(input, indices);
    return habana_lazy::empty_hpu_lazy(
        shape, input.options(), input.suggest_memory_format(), false);
  }
}
} // namespace habana
