//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
// All Rights Reserved.
//
// Unauthorized copying of this file or any element(s) within it, via any medium
// is strictly prohibited.
// This file contains Habana Labs, Ltd. proprietary and confidential information
// and is subject to the confidentiality and license agreements under which it
// was provided.
//
//===----------------------------------------------------------------------===//

#include "generated/lazy/kthvalue.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

template <>
LazyKthvalue<std::tuple<at::Tensor, at::Tensor>>::LazyKthvalue(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
std::tuple<at::Tensor, at::Tensor> LazyKthvalue<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto self = get_inputs().at(0).toTensor();
  at::Tensor values = habana_lazy::empty_hpu_lazy(
      get_out_shapes()[0], self.options(), self.suggest_memory_format(), false);
  at::Tensor indices = habana_lazy::empty_hpu_lazy(
      get_out_shapes()[1],
      self.options().dtype(c10::ScalarType::Long),
      self.suggest_memory_format(),
      false);
  return {values, indices};
}

} // namespace habana