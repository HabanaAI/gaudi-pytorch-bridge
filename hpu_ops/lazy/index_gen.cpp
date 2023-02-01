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
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

FALLBACK_CHECK(
    IndexFallbackCheck,
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

template <>
LazyIndex<at::Tensor>::LazyIndex(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  habana_lazy::NoAccThread no_acc_thread;

  auto& sub_inputs = get_inputs();
  const at::Tensor self = sub_inputs.at(0).toTensor();
  c10::ArrayRef<c10::IValue> indices_in = sub_inputs.at(1).toListRef();
  std::vector<at::Tensor> indices_vec_out{};
  std::vector<at::Tensor> indices_vec;
  for (auto input : indices_in) {
    auto o1 = input.toOptional<at::Tensor>();

    if (!(o1.has_value() && !o1->defined())) {
      indices_vec.push_back(o1.value());
    } else {
      HABANA_ASSERT(
          0 &&
          "None is not yet supported on HPU for c10::List<c10::optional<Tensor>>");
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
  if (indices_vec[0].scalar_type() == c10::ScalarType::Bool) {
    for (size_t i = 0; i < indices_vec.size(); i++) {
      auto list = torch::nonzero_numpy(indices_vec.at(i));
      indices_vec_out.insert(
          indices_vec_out.cend(), list.cbegin(), list.cend());
    }
  }

  at::TensorList indices =
      (indices_vec[0].scalar_type() == c10::ScalarType::Bool) ? indices_vec_out
                                                              : indices_vec;

  auto indices_out_vec =
      habana_lazy::HbLazyTensorViews::HandleViewsTensorList(indices);
  at::TensorList indices_out_list(indices_out_vec);

  get_inputs().back() = indices_out_list;
}

template <>
at::Tensor LazyIndex<at::Tensor>::get_result_overrideable() {
  auto inputs = get_inputs();
  const at::Tensor input = inputs[0].toTensor();
  auto indices = inputs[1].toTensorList().vec();

  auto shape = IndexOperator::compute_output_shape(input, indices);
  return habana_lazy::empty_hpu_lazy(
      shape, input.options(), input.suggest_memory_format(), false);
}
} // namespace habana
