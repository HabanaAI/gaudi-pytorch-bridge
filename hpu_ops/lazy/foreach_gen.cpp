/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/lazy/_foreach_acos.h"
#include "generated/lazy/_foreach_add.h"
#include "generated/lazy/_foreach_exp.h"
#include "generated/lazy/_foreach_zero.h"

namespace habana {

template <>
ForeachFE<::std::vector<at::Tensor>>::ForeachFE(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<::std::vector<at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
::std::vector<at::Tensor> ForeachFE<
    ::std::vector<at::Tensor>>::get_result_overrideable() {
  ::std::vector<at::Tensor> tensors;
  for (const auto& tensor : get_inputs()[0].toTensorList()) {
    tensors.emplace_back(at::empty_like(tensor));
  }
  return tensors;
}

template <>
ForeachBinaryFE<::std::vector<at::Tensor>>::ForeachBinaryFE(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<::std::vector<at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
::std::vector<at::Tensor> ForeachBinaryFE<
    ::std::vector<at::Tensor>>::get_result_overrideable() {
  ::std::vector<at::Tensor> tensors;
  const auto& stack = get_inputs();
  const auto& list1 = stack.at(0).toTensorList();

  if (stack.at(1).isList()) {
    const auto& list2 = stack.at(1).toList();
    TORCH_CHECK(
        list1.size() == list2.size(),
        "List1 size: ",
        list1.size(),
        ", != List2 size: ",
        list2.size());
    for (auto i = 0u; i < list1.size(); ++i) {
      at::Tensor t1 = list1[i];
      at::ScalarType dtype;
      std::vector<int64_t> sizes;
      if (list2[i].isTensor()) {
        at::Tensor t2 = list2[i].toTensor();
        dtype = at::result_type(t1, t2);
        sizes = at::infer_size(t1.sizes(), t2.sizes());
      } else {
        dtype = at::result_type(t1, list2[i].toScalar());
        sizes = t1.sizes().vec();
      }
      tensors.emplace_back(habana_lazy::empty_hpu_lazy(
          sizes, dtype, t1.suggest_memory_format()));
    }
  } else {
    const auto& scalar = stack.at(1).toScalar();
    for (const auto& t : list1) {
      auto dtype = at::result_type(t, scalar);
      tensors.emplace_back(at::empty_like(t, dtype));
    }
  }
  return tensors;
}
} // namespace habana
