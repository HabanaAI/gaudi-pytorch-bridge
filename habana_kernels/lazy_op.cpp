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
    bool is_outfn,
    bool safe_cast_check,
    const std::function<std::vector<std::vector<int64_t>>(const at::Stack&)>&
        out_shapes_fn)
    : LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  HABANA_ASSERT(!is_outfn, "Unexpected output op variant");
  dtype_helper_ = habana_helpers::DTypeHelper::binary_op_with_type_promotion(
      inputs, c10::nullopt, safe_cast_check);
}

template <>
LazyOpWithTypePromotion<at::Tensor&>::LazyOpWithTypePromotion(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    bool is_outfn,
    bool safe_cast_check,
    const std::function<std::vector<std::vector<int64_t>>(const at::Stack&)>&
        out_shapes_fn)
    : LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Perform type promotion and validate if promoted type can be casted to
  // output data type.
  auto output = c10::make_optional<const at::IValue*>(
      is_outfn ? &inputs.back() : &inputs.front());
  dtype_helper_ = habana_helpers::DTypeHelper::binary_op_with_type_promotion(
      inputs, output, safe_cast_check);
}

template <>
at::Tensor LazyOpWithTypePromotion<at::Tensor>::get_result_overrideable() {
  const auto& inputs = LazyOp<at::Tensor>::get_inputs();
  at::Tensor t;
  auto result_type = dtype_helper_.get_result_dtype();

  t = inputs.at(0).isTensor() ? inputs.at(0).toTensor()
                              : inputs.at(1).toTensor();

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
    bool /*is_outfn*/,
    bool /*safe_cast_check*/,
    const std::function<std::vector<std::vector<int64_t>>(const at::Stack&)>&
        out_shapes_fn)
    : LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  set_scalar_type(habana_helpers::DTypeHelper::get_compute_dtype(
      inputs, c10::nullopt, true, true, false));
}

template <>
PromoteIntToFloat<at::Tensor&>::PromoteIntToFloat(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    bool is_outfn,
    bool /*safe_cast_check*/,
    const std::function<std::vector<std::vector<int64_t>>(const at::Stack&)>&
        out_shapes_fn)
    : LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Perform type promotion and validate if promoted type can cast to
  // output data type.
  auto&& output = c10::make_optional<const at::IValue*>(
      is_outfn ? &inputs.back() : &inputs.front());
  // Exclude out tensor from inputs
  at::Stack op_inputs = inputs;
  if (is_outfn) {
    op_inputs = {inputs.begin(), inputs.end() - 1};
  }
  set_scalar_type(habana_helpers::DTypeHelper::get_compute_dtype(
      op_inputs, output, true, true, true));
}

template <>
at::Tensor PromoteIntToFloat<at::Tensor>::get_result_overrideable() {
  return LazyOp<at::Tensor>::get_result_overrideable();
}

template <>
at::Tensor& PromoteIntToFloat<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

template <>
at::Tensor LazyBinaryOp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = LazyOp<at::Tensor>::get_inputs();
  const auto& self = inputs.at(0).toTensor();

  const auto& outshape = LazyOp<at::Tensor>::get_out_shapes().empty()
      ? self.sizes()
      : LazyOp<at::Tensor>::get_out_shapes().at(0);

  if (dst_dtype_ == at::ScalarType::Undefined) {
    c10::optional<const at::IValue*> output = is_outfn_
        ? c10::make_optional<const at::IValue*>(&inputs.back())
        : c10::nullopt;
    auto dtype_helper =
        habana_helpers::DTypeHelper::binary_op_with_type_promotion(
            inputs, output, safe_cast_check_);

    dst_dtype_ = dtype_helper.get_result_dtype();
  }

  return empty_hpu_lazy(
      outshape,
      self.options().device(c10::kHPU).dtype(dst_dtype_),
      self.suggest_memory_format(),
      false);
}

template <>
at::Tensor& LazyBinaryOp<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

} // namespace habana_lazy
