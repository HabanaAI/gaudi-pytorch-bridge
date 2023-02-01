/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/lazy/bernoulli.h"
#include "generated/lazy/poisson.h"
#include "generated/lazy/random.h"
#include "generated/lazy/uniform.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

template <typename T>
LazyTensorSeed<T>::LazyTensorSeed(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyTensorSeed<T>::get_inputs().back() =
      get_seed_tensor_hpu(inputs.back().toOptional<at::Generator>());
}

template <typename T>
T LazyTensorSeed<T>::get_result_overrideable() {
  return stack_tensor(LazyTensorSeed<T>::get_inputs(), 0);
}

template struct LazyTensorSeed<at::Tensor&>;
template struct LazyTensorSeed<at::Tensor>;

template <typename T>
LazyTensorOutSeed<T>::LazyTensorOutSeed(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyTensorOutSeed<T>::get_inputs().at(1) =
      get_seed_tensor_hpu(inputs.at(1).toOptional<at::Generator>());
}

template <typename T>
T LazyTensorOutSeed<T>::get_result_overrideable() {
  return stack_tensor(LazyTensorOutSeed<T>::get_inputs(), 0);
}

template struct LazyTensorOutSeed<at::Tensor&>;
template struct LazyTensorOutSeed<at::Tensor>;
} // namespace habana
