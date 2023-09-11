/*******************************************************************************
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
#include <ATen/InferSize.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/SymIntArrayRef.h>

#include "backend/backend_meta.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/index_put.h"
#include "hpu_ops/cpu_fallback.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_logger.h"

#define MAX_DIMS_FOR_ADVANCED_INDEXING (8)

namespace habana {
namespace eager {

static bool check_for_advanced_indexing(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  bool advanced_indexing = false;
  c10::ScalarType prev_scalar_type = c10::ScalarType::Long;
  bool first_scalar = true;

  if (indices.size() <= MAX_DIMS_FOR_ADVANCED_INDEXING) {
    for (c10::optional<at::Tensor> input_ind : indices) {
      auto input = input_ind.value_or(at::Tensor());
      if (!input.defined()) {
        advanced_indexing = true;
        break;
      } else {
        // if we are indexing using a mixture of long and boolean indices,then
        // also we will work in advanced indexing mode
        auto cur_scalar_type = input.scalar_type();
        if (first_scalar) {
          first_scalar = false;
          prev_scalar_type = cur_scalar_type;
        } else {
          if (prev_scalar_type != cur_scalar_type) {
            advanced_indexing = true;
            break;
          }
        }
        prev_scalar_type = cur_scalar_type;
      }
    }
  }
  return advanced_indexing;
}

static c10::List<c10::optional<at::Tensor>> check_for_boolean_advanced_indexing(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  std::vector<c10::optional<at::Tensor>> bool_indices_vec;
  at::Tensor t_nz;
  bool has_bool_mask = false;
  for (c10::optional<at::Tensor> input_ind : indices) {
    auto input_temp = input_ind.value_or(at::Tensor());
    at::Tensor input;
    if (input.defined() &&
        (input_temp.device().type() != c10::DeviceType::HPU)) {
      input = input_temp.to(c10::kHPU);
    } else {
      input = input_temp;
    }
    if (!input.defined()) {
      bool_indices_vec.emplace_back(input);
    } else {
      auto cur_scalar_type = input.scalar_type();
      if (cur_scalar_type == c10::ScalarType::Bool) {
        has_bool_mask = true;
        auto nonzero_indices = at::nonzero(input);
        t_nz = at::squeeze(nonzero_indices, 1);
        if (t_nz.dim() > 1) {
          std::vector<int64_t> dims_sz_vec(t_nz.sizes()[1], 1);
          c10::IntArrayRef dims_sz(dims_sz_vec);
          auto nz_indices = at::split_with_sizes(t_nz, dims_sz, 1);
          for (auto i : c10::irange((int)nz_indices.size())) {
            auto nzi = at::squeeze(nz_indices.at(i), 1);
            bool_indices_vec.emplace_back(nzi);
          }
        } else {
          bool_indices_vec.emplace_back(t_nz);
        }
      } else {
        bool_indices_vec.emplace_back(input);
      }
    }
  }
  if (has_bool_mask) {
    c10::List<c10::optional<at::Tensor>> bool_mask_indices(bool_indices_vec);
    return bool_mask_indices;
  } else {
    return indices;
  }
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
static C10_UNUSED std::tuple<at::Tensor, std::vector<c10::optional<at::Tensor>>>
transposeToFront(const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();
  std::vector<int64_t> dims;
  std::vector<c10::optional<at::Tensor>> transposedIndices;
  std::vector<c10::optional<at::Tensor>> indices;
  for (const auto& index_opt : indices_ival) {
    auto o1 = index_opt.toOptional<at::Tensor>();
    if (o1.has_value() && o1.value().defined()) {
      const auto& index = o1.value();
      indices.emplace_back(std::move(index));
    } else {
      indices.emplace_back(c10::nullopt);
    }
  }
  dims.reserve(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].has_value()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].has_value()) {
      dims.push_back(i);
      transposedIndices.emplace_back(c10::nullopt);
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

static std::tuple<at::Tensor, std::vector<at::Tensor>>
generate_advanced_indexing_indices_list(const at::Stack& stack) {
  at::Tensor self = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();

  std::vector<c10::optional<at::Tensor>> indices;
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  for (const auto& index_opt : indices_ival) {
    auto o1 = index_opt.toOptional<at::Tensor>();
    if (o1.has_value() && o1.value().defined()) {
      const auto& index = o1.value();
      indices.emplace_back(std::move(index));
    } else {
      indices.emplace_back(c10::nullopt);
    }
  }
  auto self_sizes = self.sizes().vec();
  std::vector<at::Tensor> indices_list;
  int64_t explicit_index_count = 0;
  int64_t broadcast_to_size = 1;
  int64_t i = 0;
  std::vector<int64_t> broadcast_to_shape;
  int64_t index_t_sizes[self.dim()];
  bool index_all_elems[self.dim()];
  for (auto index_input : indices) {
    auto input = index_input;
    if (input.has_value() &&
        (input.value().scalar_type() != c10::ScalarType::Bool)) {
      auto cur_index_dim_size = input.value().sizes().vec()[0];
      if (cur_index_dim_size >= broadcast_to_size) {
        broadcast_to_size = cur_index_dim_size;
        broadcast_to_shape = input.value().sizes().vec();
      }
      index_all_elems[i] = false;
      index_t_sizes[i] = input.value().sizes()[0];
      explicit_index_count++;
    } else if (!input.has_value()) {
      index_t_sizes[i] = self_sizes[i];
      index_all_elems[i] = true;
    }
    i++;
  }
  // account for any trailing dims that are not specified to be
  // indexed explicitly, but need to be taken care of.
  for (; i < self.dim(); i++) {
    index_all_elems[i] = true;
  }
  int64_t repeats_needed[self.dim()];
  int64_t repeat_interleaves_needed[self.dim()];
  for (i = 0; i < self.dim(); i++) {
    repeats_needed[i] = 1;
    int64_t total_elements_above = 1;
    bool explicit_index_above = false;
    for (int j = 0; j < i; j++) {
      total_elements_above *= (index_all_elems[j]) ? self_sizes[j] : 1;
      if (!index_all_elems[j]) {
        explicit_index_above = true;
      }
    }
    repeats_needed[i] = total_elements_above;
    if (index_all_elems[i] && explicit_index_above) {
      repeats_needed[i] *= broadcast_to_size;
    } else if (!index_all_elems[i] && (1 == index_t_sizes[i])) {
      repeats_needed[i] *= broadcast_to_size;
    }

    repeat_interleaves_needed[i] = 1;
    int64_t total_elements_below = 1;
    bool explicit_index_below = false;
    for (int j = i + 1; j < self.dim(); j++) {
      total_elements_below *= (index_all_elems[j]) ? self_sizes[j] : 1;
      if (!index_all_elems[j]) {
        explicit_index_below = true;
      }
    }
    if (index_all_elems[i] && explicit_index_below && !explicit_index_above) {
      total_elements_below *= broadcast_to_size;
    }
    repeat_interleaves_needed[i] = total_elements_below;
  }

  // Now create all indices tensors and insert into list using repeats_needed
  // and repeat_interleaves_needed. Note that in Boolean mask indexing method we
  // have to create indices from the mask using nonzero and squeeze as done in
  // previous code block.
  for (int dim = 0; dim < self.dim(); dim++) {
    at::Tensor it;
    if (index_all_elems[dim]) {
      std::vector<int64_t> shape{self.sizes().vec()[dim]};
      c10::IntArrayRef arange_size(shape.data(), shape.size());
      at::TensorOptions options =
          self.options().dtype(c10::ScalarType::Long).device(c10::kHPU);
      auto generated_index_tensor =
          at::empty(arange_size, options, c10::nullopt);
      generated_index_tensor = at::arange(
          0,
          self.sizes().vec()[dim],
          1,
          c10::ScalarType::Long,
          c10::nullopt,
          c10::kHPU,
          c10::nullopt);
      it = generated_index_tensor;
      auto it_repeat_interleave =
          it.repeat_interleave(repeat_interleaves_needed[dim]);
      indices_list.push_back(it_repeat_interleave.repeat(repeats_needed[dim]));
    } else if (indices[dim].has_value()) {
      auto input_temp = indices[dim].value();
      at::Tensor in_t;
      if (input_temp.defined() &&
          (input_temp.device().type() != c10::DeviceType::HPU)) {
        in_t = input_temp.to(c10::kHPU);
      } else {
        in_t = input_temp;
      }
      auto it_repeat_interleave =
          in_t.repeat_interleave(repeat_interleaves_needed[dim]);
      indices_list.push_back(it_repeat_interleave.repeat(repeats_needed[dim]));
    }
  }
  return std::make_tuple(self, std::move(indices_list));
}

at::Tensor& _index_put_impl_eager(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices_in,
    const at::Tensor& value,
    bool accumulate,
    bool unsafe) {
  c10::List<c10::optional<at::Tensor>> indices;
  bool advanced_indexing = check_for_advanced_indexing(indices_in);
  if (advanced_indexing) {
    // if we have boolean mask tensors, convert them to long int indices
    indices = check_for_boolean_advanced_indexing(indices_in);
  } else {
    indices = indices_in;
  }
  std::vector<at::Tensor> indices_vec;
  TORCH_CHECK(
      self.dim() <= MAX_DIMS_FOR_ADVANCED_INDEXING,
      "index_put op doesn't support more than ",
      MAX_DIMS_FOR_ADVANCED_INDEXING,
      " dims");
  at::Tensor self_permuted;
  if (self.device().type() != c10::DeviceType::HPU)
    self_permuted = self.to(c10::kHPU);
  else
    self_permuted = self;

  at::Tensor value_in;
  if (value.device().type() != c10::DeviceType::HPU)
    value_in = value.to(c10::kHPU);
  else
    value_in = value;
  if (advanced_indexing) {
    at::Stack stack;
    stack.emplace_back(self);
    stack.emplace_back(c10::IValue(indices));
    std::tie(self_permuted, indices_vec) =
        generate_advanced_indexing_indices_list(stack); //(self, indices_in);
  } else {
    for (c10::optional<at::Tensor> input_ind : indices) {
      auto input_temp = input_ind.value_or(at::Tensor());
      at::Tensor input;
      if (input_temp.defined() &&
          (input_temp.device().type() != c10::DeviceType::HPU)) {
        input = input_temp.to(c10::kHPU);
      } else {
        input = input_temp;
      }
      if (input.defined()) {
        indices_vec.push_back(input);
      } else {
        HABANA_ASSERT(
            0 &&
            "index_put: unsupported case: None is not yet supported on HPU for c10::List<c10::optional<Tensor>>");
      }
    }
  }
  std::vector<c10::optional<at::Tensor>> indices_out_opt_vec;
  for (auto ind : indices_vec) {
    if (ind.defined()) {
      indices_out_opt_vec.emplace_back(ind);
    } else {
      indices_out_opt_vec.emplace_back(c10::nullopt);
    }
  }

  indices_vec.clear();
  for (c10::optional<at::Tensor> input : indices_out_opt_vec) {
    indices_vec.push_back(input.value());
  }

  const bool areAllIndicesBool =
      std::all_of(indices_vec.cbegin(), indices_vec.cend(), [](const auto& i) {
        return i.scalar_type() == c10::ScalarType::Bool;
      });
  auto only_single_index_tensor = ((int)indices_vec.size() == 1) ? true : false;
  auto self_clone = self;
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }
  // For ZST indices tensor scatter_nd
  // operation is throwing GC error therefore we have this workaround to
  // return a copy of input tensor.
  // GC Jira - SW-73941
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].numel() == 0 || value_in.numel() == 0) {
      return self;
    }
  }
  std::vector<at::Tensor> indices_vec_out{};
  at::Tensor nzt;

  if (areAllIndicesBool &&
      (advanced_indexing || !only_single_index_tensor ||
       !GET_ENV_FLAG_NEW(PT_HPU_EAGER_INDEX_PUT_BOOL_OPTIMIZED))) {
    for (size_t i = 0; i < indices_vec.size(); i++) {
      auto ind = indices_vec.at(i);
      at::Tensor nz = ind.nonzero();
      if (!nz.dim()) { // bool mask has all False entries
        return self;
      } else {
        auto expanded_ind = indices_vec.at(i).expand_as(self);
        nzt = at::nonzero(expanded_ind);
      }
      indices_vec_out.emplace_back(nzt);
    }
  }

  at::TensorList indices_final =
      (areAllIndicesBool &&
       (advanced_indexing || !only_single_index_tensor ||
        !GET_ENV_FLAG_NEW(PT_HPU_EAGER_INDEX_PUT_BOOL_OPTIMIZED)))
      ? indices_vec_out
      : indices_vec;
  at::Tensor result;
  if (areAllIndicesBool && !advanced_indexing && only_single_index_tensor &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_INDEX_PUT_BOOL_OPTIMIZED)) {
    habana::eager::EagerOp<at::Tensor> hpu_op{
        "hpu::_index_put_impl_bool_eager",
        {self, indices_final, value, accumulate}};
    result = hpu_op.call();
  } else {
    habana::eager::EagerOp<at::Tensor> hpu_op{
        "hpu::_index_put_impl_eager", {self, indices_final, value, accumulate}};
    result = hpu_op.call();
  }
  self.copy_(result);
  return self;
}
TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "_index_put_impl_eager(Tensor self, Tensor[] indices, Tensor value, bool accumulate=False) -> Tensor");
}
TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def(
      "_index_put_impl_bool_eager(Tensor self, Tensor[] indices, Tensor value, bool accumulate=False) -> Tensor");
}
} // namespace eager
} // namespace habana
