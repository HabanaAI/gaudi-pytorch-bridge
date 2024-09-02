/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_device/HPUEvent.h"
#include "generated/eager/index.h"
#include "generated/eager/wrap_kernels_declarations.h"
#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/common/index.h"
#include "hpu_ops/indexing_ops_helper.h"

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
        (o1.value().device() == torch::kCPU))
      return false;
  }
  return true;
};

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, IndexOutFE, at::Tensor&) {
  m_symbol = at::Symbol::fromQualString("hpu::index");
  auto& sub_inputs = get_inputs();
  const at::Tensor self = sub_inputs.at(0).toTensor();
  c10::ArrayRef<c10::IValue> indices_in_orig = sub_inputs.at(1).toListRef();
  std::vector<at::IValue> inputs_vec = sub_inputs;
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
  std::vector<bool> advanced_indexing_present;
  int dim = 0;
  at::Tensor t_nz;
  bool has_bool_mask = false;
  int num_index_tensors = (int)indices_in_orig.size();

  advanced_indexing = check_for_adv_indexing(indices_in_orig);
  has_bool_mask = handle_bool_mask_indices(
      indices_in_orig, indices_in_ivals_vec, bool_indices_vec);
  if (has_bool_mask) {
    indices_in = indices_in_ivals_vec;
    c10::List<c10::optional<at::Tensor>> bool_mask_indices(bool_indices_vec);
    inputs_vec.clear();
    inputs_vec.emplace_back(sub_inputs.at(0));
    inputs_vec.emplace_back(c10::IValue(bool_mask_indices));
  } else {
    indices_in = indices_in_orig;
  }

  at::Tensor self_permuted = self;
  std::vector<int64_t> self_permute_dims;
  if (advanced_indexing) {
    // If indices for the given dim are not initialized or their values
    // are not defined then populate the advanced_indexing_present with
    // true else false. The case with dims_permuted triggers when there
    // is more than one group of explicit indices detected e.g.
    // ([0, 1], None, [0, 2]), such indices are transposed to the front then.
    bool explicit_indices_together = false;
    int index_tensor_group_start = 0;
    int index_tensor_group_end = 0;
    int num_explicit_indices = 0;

    for (auto input : indices_in) {
      auto o1 = input.toOptional<at::Tensor>();
      if (o1.has_value() && o1.value().defined()) {
        if (!explicit_indices_together) {
          explicit_indices_together = true;
          index_tensor_group_start = dim;
        }
        index_tensor_group_end = dim;
        num_explicit_indices++;
      } else {
        explicit_indices_together = false;
      }
      dim++;
    }

    bool dims_permuted = false;
    std::tie(dims_permuted, num_index_tensors, self_permute_dims, indices_vec) =
        generate_advanced_indexing_indices_list(inputs_vec);
    if (dims_permuted)
      for (const auto i : c10::irange(num_index_tensors))
        advanced_indexing_present.emplace_back(i >= num_explicit_indices);
    else if (num_explicit_indices > 0)
      for (const auto i : c10::irange(num_index_tensors))
        advanced_indexing_present.emplace_back(
            (i < index_tensor_group_start) || (i > index_tensor_group_end));

    for (int i = num_index_tensors; i < (int)indices_in.size(); i++)
      advanced_indexing_present.emplace_back(true);

  } else { // advanced indexing end
    for (auto input : indices_in) {
      auto o1 = input.toOptional<at::Tensor>();
      if (o1.has_value() && o1->defined()) {
        indices_vec.push_back(o1.value());
      }
      advanced_indexing_present.emplace_back(false);
    }
    for (int i = 0; i < (int)self.dim(); i++) {
      self_permute_dims.emplace_back(i);
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

  int orig_in_tensor_type_count = (int)get_inputs().size();
  get_inputs().resize(get_inputs().size() + 3);
  get_inputs().at(0) = c10::IValue(self_permuted);
  get_inputs().at(1) = c10::IValue(indices_out_list);
  if (3 == orig_in_tensor_type_count) { // out variant
    auto& sub_inputs = get_inputs();
    auto out = sub_inputs.at(2).toTensor();
    get_inputs().at(2) = advanced_indexing_present;
    get_inputs().at(3) = self_permute_dims;
    get_inputs().at(4) = num_index_tensors;
    get_inputs().at(5) = out;
  } else {
    get_inputs().at(2) = advanced_indexing_present;
    get_inputs().at(3) = self_permute_dims;
    get_inputs().at(4) = num_index_tensors;
  }
}
} // namespace habana
