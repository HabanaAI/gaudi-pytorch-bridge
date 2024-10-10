/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/lazy/special_xlog1py.h"
#include "generated/lazy/xlogy.h"

namespace habana {

constexpr size_t index_of_self = 0;
constexpr size_t index_of_other = 1;

static void XlogyScalarConversion(
    std::vector<at::IValue>& inputs,
    size_t scalar_index,
    size_t tensor_index) {
  auto tensor = inputs[tensor_index].toTensor();
  auto scalar = inputs[scalar_index].toScalar();
  auto dtype = at::result_type(tensor, scalar);
  auto self_tensor =
      habana_lazy::get_tensor_for_scalar(scalar.toDouble(), dtype);
  inputs[scalar_index] = c10::IValue(self_tensor);
}

template <typename T>
LazyXlogY<T>::LazyXlogY(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  auto x = LazyXlogY<T>::get_inputs();
  // convert scalar input to tensor
  if (x[index_of_self].isScalar()) {
    XlogyScalarConversion(x, index_of_self, index_of_other);
  } else {
    XlogyScalarConversion(x, index_of_other, index_of_self);
  }
  LazyXlogY<T>::set_inputs(x);
}

template struct LazyXlogY<at::Tensor&>;
template struct LazyXlogY<at::Tensor>;

template <typename T>
T LazyXlogY<T>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return habana_lazy::LazyOp<T>::get_result_overrideable();
}
} // namespace habana
