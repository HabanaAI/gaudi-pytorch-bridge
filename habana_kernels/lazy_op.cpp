/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "lazy_kernels.h" // TODO rename to lazy_op.h

namespace habana_lazy {

template <>
LazyOpWithTypePromotion<at::Tensor>::LazyOpWithTypePromotion(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<
        std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
        out_shapes_fn) noexcept
    : LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
LazyOpWithTypePromotion<at::Tensor&>::LazyOpWithTypePromotion(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<
        std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
        out_shapes_fn) noexcept
    : LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {}

template <>
at::Tensor LazyOpWithTypePromotion<at::Tensor>::get_result_overrideable() {
  const auto& inputs = LazyOp<at::Tensor>::get_inputs();
  const auto& self = inputs.at(0).toTensor();
  at::ScalarType result_type;

  if (inputs.at(1).isTensor()) {
    result_type = at::result_type(self, inputs.at(1).toTensor());
  } else {
    result_type = at::result_type(self, inputs.at(1).toScalar());
  }

  const auto& outshape = LazyOp<at::Tensor>::get_out_shapes().empty()
      ? self.sizes()
      : LazyOp<at::Tensor>::get_out_shapes().at(0);

  return empty_hpu_lazy(
      outshape,
      self.options().device(c10::kHPU).dtype(result_type),
      self.suggest_memory_format(),
      false);
}

template <>
at::Tensor& LazyOpWithTypePromotion<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}
} // namespace habana_lazy
