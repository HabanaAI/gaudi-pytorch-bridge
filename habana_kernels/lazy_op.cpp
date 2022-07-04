/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_helpers/dtype_helpers.h"
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

  habana_helpers::DTypeHelper dtype_helper;
  dtype_helper.add_inputs({&inputs.at(0), &inputs.at(1)})
      .set_promote_to_common_type(true)
      .build();
  at::ScalarType result_type = dtype_helper.get_result_dtype();

  if (inputs.at(0).isTensor()) {
    t = inputs.at(0).toTensor();
  } else {
    t = inputs.at(1).toTensor();
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

  habana_helpers::DTypeHelper dtype_helper;
  dtype_helper.add_inputs({&inputs.at(0), &inputs.at(1)})
      .set_promote_to_common_type(true)
      .set_promote_int_to_float(true)
      .build();
  at::ScalarType result_type = dtype_helper.get_result_dtype();

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
