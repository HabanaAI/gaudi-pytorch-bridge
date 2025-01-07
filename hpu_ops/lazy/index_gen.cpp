/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "generated/lazy/gather.h"
#include "generated/lazy/index.h"
#include "hpu_ops/common/index.h"
#include "hpu_ops/indexing_ops_helper.h"

namespace habana {

FALLBACK_CHECK(
    IndexFallbackCheck,
    [[maybe_unused]] const c10::List<c10::optional<at::Tensor>>& indices) {
  return true;
};

static inline void index_fe(torch::jit::Stack& in_stack) {
  auto& sub_inputs = in_stack;
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

  std::vector<bool> advanced_indexing_present;
  int dim = 0;
  at::Tensor t_nz;
  bool has_bool_mask = false;
  int num_index_tensors = (int)indices_in_orig.size();
  bool advanced_indexing = false;
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

  } else { // advanced_indexing end
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
  int orig_in_tensor_type_count = (int)in_stack.size();
  in_stack.resize(in_stack.size() + 3);
  in_stack.at(0) = self_permuted;
  in_stack.at(1) = indices_out_list;
  if (3 == orig_in_tensor_type_count) { // out variant
    auto& sub_inputs = in_stack;
    auto out = sub_inputs.at(2).toTensor();
    in_stack.at(2) = advanced_indexing_present;
    in_stack.at(3) = self_permute_dims;
    in_stack.at(4) = num_index_tensors;
    in_stack.at(5) = out;
  } else {
    in_stack.at(2) = advanced_indexing_present;
    in_stack.at(3) = self_permute_dims;
    in_stack.at(4) = num_index_tensors;
  }
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, IndexFE, at::Tensor) {
  habana_lazy::NoAccThread no_acc_thread;
  index_fe(get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, IndexOutFE, at::Tensor&) {
  index_fe(get_inputs());
}

} // namespace habana
