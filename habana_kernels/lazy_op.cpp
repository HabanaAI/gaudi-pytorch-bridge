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
  at::Tensor t;
  at::ScalarType result_type;

  if (inputs.at(0).isTensor()) {
    t = inputs.at(0).toTensor();
    if (inputs.at(1).isTensor()) {
      result_type = at::result_type(t, inputs.at(1).toTensor());
    } else {
      result_type = at::result_type(t, inputs.at(1).toScalar());
    }
  } else {
    t = inputs.at(1).toTensor();
    result_type = at::result_type(inputs.at(0).toScalar(), t);
  }

  const auto& outshape = LazyOp<at::Tensor>::get_out_shapes().empty()
      ? t.sizes()
      : LazyOp<at::Tensor>::get_out_shapes().at(0);

  return empty_hpu_lazy(
      outshape,
      t.options().device(c10::kHPU).dtype(result_type),
      t.suggest_memory_format(),
      false);
}

template <>
at::Tensor& LazyOpWithTypePromotion<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

static c10::ScalarType get_promoted_float_type(
    const at::Tensor& t,
    at::Scalar s) {
  if (c10::isIntegralType(t.scalar_type(), true) and s.isIntegral(true)) {
    return at::get_default_dtype_as_scalartype();
  }
  return at::result_type(t, s);
}

static c10::ScalarType get_promoted_float_type(
    const at::Tensor& t1,
    const at::Tensor& t2) {
  if (c10::isIntegralType(t1.scalar_type(), true) and
      c10::isIntegralType(t2.scalar_type(), true)) {
    return at::get_default_dtype_as_scalartype();
  }
  return at::result_type(t1, t2);
}

template <>
PromoteIntToFloat<at::Tensor>::PromoteIntToFloat(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<
        std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
        out_shapes_fn) noexcept
    : LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
PromoteIntToFloat<at::Tensor&>::PromoteIntToFloat(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<
        std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
        out_shapes_fn) noexcept
    : LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {}

template <>
at::Tensor PromoteIntToFloat<at::Tensor>::get_result_overrideable() {
  const auto& inputs = LazyOp<at::Tensor>::get_inputs();
  const auto& self = inputs.at(0).toTensor();
  at::ScalarType result_type = at::get_default_dtype_as_scalartype();

  if (inputs.at(1).isTensor()) {
    result_type = get_promoted_float_type(self, inputs.at(1).toTensor());
  } else {
    result_type = get_promoted_float_type(self, inputs.at(1).toScalar());
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
at::Tensor& PromoteIntToFloat<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}
} // namespace habana_lazy
