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

#include <c10_ver/core/SymIntArrayRef.h>
#include "generated/eager/index.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
#include "hpu_ops/common/index.h"
#include "hpu_ops/indexing_ops_helper.h"
#include "pytorch_helpers/habana_device/HPUEvent.h"

namespace habana {

FALLBACK_CHECK(
    IndexFallbackCheck,
    [[maybe_unused]] const c10::List<c10::optional<at::Tensor>>& indices) {
  at::Stack stack = {indices};
  c10::ArrayRef<c10::IValue> indices_in = stack.at(0).toListRef();
  // TBD: NOTE: For eager: we are going to execute on CPU if indices are either
  // boolean or they are on CPU
  for (auto input : indices_in) {
    auto o1 = input.toOptional<at::Tensor>();
    if (o1.has_value() && o1.value().defined() &&
        o1.value().device() == torch::kCPU) {
      return false;
    } else if (
        o1.has_value() && o1.value().defined() &&
        (o1.value().scalar_type() != c10::ScalarType::Bool)) {
      continue;
    } else if (
        o1.has_value() && (o1.value().scalar_type() == c10::ScalarType::Bool)) {
      return false;
    }
  }
  return true;
};

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, IndexFE, at::Tensor) {
  TORCH_CHECK(
      0, "IndexFE is not expected to be called for PT 2.0 as i is DFDT");
}

HPU_OP_FRONTEND_CUSTOM_CTOR(eager::EagerOp, IndexOutFE, -1, at::Tensor&) {
  auto& sub_inputs = get_inputs();
  const at::Tensor self = sub_inputs.at(0).toTensor();
  c10::ArrayRef<c10::IValue> indices_in_orig = sub_inputs.at(1).toListRef();
  std::vector<at::IValue> inputs_vec = sub_inputs; // inputs_orig;
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
      -1, -1, -1, -1, -1};
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

  advanced_indexing = check_for_adv_indexing(indices_in_orig);

  if (advanced_indexing) {
    has_bool_mask = handle_bool_mask_indices(
        indices_in_orig, indices_in_ivals_vec, bool_indices_vec);
  }
  if (advanced_indexing && has_bool_mask) {
    indices_in = indices_in_ivals_vec;
    c10::List<c10::optional<at::Tensor>> bool_mask_indices(bool_indices_vec);
    inputs_vec.clear();
    inputs_vec.emplace_back(sub_inputs.at(0));
    inputs_vec.emplace_back(c10::IValue(bool_mask_indices));
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
        generate_advanced_indexing_indices_list(inputs_vec);
    if ((index_tensor_groups > 1) &&
        (num_explicit_indices >
         1)) { // all explicitly indexed dims are now in higher order dims.
      for (const auto i : c10::irange((int)implicit_indices_pos_vec.size())) {
        if (i < num_explicit_indices) {
          advanced_indexing_dims[i] = implicit_indices_pos_vec[i];
        } else {
          advanced_indexing_dims[i] = 0;
        }
      }
    } else if (num_explicit_indices > 1) {
      for (const auto i : c10::irange((int)implicit_indices_pos_vec.size())) {
        if ((i >= index_tensor_group_start) && (i <= index_tensor_group_end)) {
          advanced_indexing_dims[i] = implicit_indices_pos_vec[i];
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
      TORCH_CHECK(0, "Indexing with CPU tensors is not supported for PT 2.0");
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }

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

  at::TensorList indices_out_list =
      (bool_non_adv_indexing_case) ? indices_vec_out : indices_vec;
  // at::TensorList indices_out_list(indices);
  int orig_in_tensor_type_count = (int)get_inputs().size();
  get_inputs().resize(get_inputs().size() + 3);
  get_inputs().at(0) = c10::IValue(self_permuted);
  get_inputs().at(1) = c10::IValue(indices_out_list);
  if (3 == orig_in_tensor_type_count) { // out variant
    auto& sub_inputs = get_inputs();
    auto out = sub_inputs.at(2).toTensor();
    get_inputs().at(2) = advanced_indexing_dims;
    get_inputs().at(3) = implicit_indices_pos_vec;
    get_inputs().at(4) = self_permute_dims;
    get_inputs().at(5) = out;
  } else {
    get_inputs().at(2) = advanced_indexing_dims;
    get_inputs().at(3) = implicit_indices_pos_vec;
    get_inputs().at(4) = self_permute_dims;
  }
}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(eager::EagerOp, IndexOutFE, at::Tensor&) {
  return get_index_result_out(get_inputs());
}

} // namespace habana
