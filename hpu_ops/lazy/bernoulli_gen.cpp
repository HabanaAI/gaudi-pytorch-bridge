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

#include "generated/lazy/bernoulli.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

template <>
LazyBernoulliOutFrontend<at::Tensor&>::LazyBernoulliOutFrontend(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyBernoulliOutFrontend<at::Tensor&>::get_inputs().at(1) =
      habana_lazy::get_tensor_for_scalar(
          inputs[1].toDouble(), inputs[0].toTensor().options());
  LazyBernoulliOutFrontend<at::Tensor&>::get_inputs().at(2) =
      get_seed_tensor_hpu(inputs.at(2).toOptional<at::Generator>());
}

template <>
at::Tensor& LazyBernoulliOutFrontend<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(LazyBernoulliOutFrontend<at::Tensor&>::get_inputs(), 0);
}

template <typename T>
LazyBernoulliFrontend<T>::LazyBernoulliFrontend(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  LazyBernoulliFrontend<T>::get_inputs().at(1) =
      habana_lazy::get_tensor_for_scalar(
          inputs[1].toDouble(), inputs[0].toTensor().options());
  LazyBernoulliFrontend<T>::get_inputs().back() =
      get_seed_tensor_hpu(inputs.back().toOptional<at::Generator>());
}
template <typename T>
T LazyBernoulliFrontend<T>::get_result_overrideable() {
  return stack_tensor(LazyBernoulliFrontend<T>::get_inputs(), 0);
}
template struct LazyBernoulliFrontend<at::Tensor&>;
template struct LazyBernoulliFrontend<at::Tensor>;
} // namespace habana
