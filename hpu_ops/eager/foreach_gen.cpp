// /******************************************************************************
//  * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
//  * All Rights Reserved.
//  *
//  * Unauthorized copying of this file or any element(s) within it, via any
//  medium
//  * is strictly prohibited.
//  * This file contains Habana Labs, Ltd. proprietary and confidential
//  information
//  * and is subject to the confidentiality and license agreements under which
//  it
//  * was provided.
//  *
//  *******************************************************************************
//  */

#include "generated/eager/_foreach_acos.h"
#include "generated/eager/_foreach_add.h"
#include "generated/eager/_foreach_exp.h"
#include "generated/eager/_foreach_zero.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR(
    eager::EagerOp,
    ForeachFE,
    -1,
    std::vector<at::Tensor>) {}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(
    eager::EagerOp,
    ForeachFE,
    std::vector<at::Tensor>) {
  ::std::vector<at::Tensor> tensors;
  for (const auto& tensor : get_inputs()[0].toTensorList()) {
    tensors.emplace_back(at::empty_like(tensor));
  }
  return tensors;
}

HPU_OP_FRONTEND_CUSTOM_CTOR(
    eager::EagerOp,
    ForeachBinaryFE,
    -1,
    std::vector<at::Tensor>) {}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(
    eager::EagerOp,
    ForeachBinaryFE,
    std::vector<at::Tensor>) {
  std::vector<at::Tensor> tensors;
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
      tensors.emplace_back(hpu_wrap::empty(
          c10::fromIntArrayRefSlow(sizes),
          dtype,
          t1.options().layout_opt(),
          t1.options().device_opt(),
          t1.options().pinned_memory_opt(),
          t1.suggest_memory_format()));
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
