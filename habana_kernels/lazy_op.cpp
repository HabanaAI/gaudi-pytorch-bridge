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
    const std::function<
        std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
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
    const std::function<
        std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
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
    bool is_outfn,
    bool safe_cast_check,
    const std::function<
        std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
        out_shapes_fn)
    : LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  HABANA_ASSERT(!is_outfn, "Unexpected output op variant");
  dtype_helper_ =
      habana_helpers::DTypeHelper::binary_op_with_int_to_float_promotion(
          inputs, c10::nullopt, safe_cast_check);
}

template <>
PromoteIntToFloat<at::Tensor&>::PromoteIntToFloat(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    bool is_outfn,
    bool safe_cast_check,
    const std::function<
        std::vector<std::vector<int64_t>>(const at::Stack&, bool)>&
        out_shapes_fn)
    : LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Perform type promotion and validate if promoted type can be casted to
  // output data type.
  auto output = c10::make_optional<const at::IValue*>(
      is_outfn ? &inputs.back() : &inputs.front());
  dtype_helper_ =
      habana_helpers::DTypeHelper::binary_op_with_int_to_float_promotion(
          inputs, output, safe_cast_check);
}

template <>
at::Tensor PromoteIntToFloat<at::Tensor>::get_result_overrideable() {
  const auto& inputs = LazyOp<at::Tensor>::get_inputs();
  const auto& self = inputs.at(0).toTensor();

  auto result_type = dtype_helper_.get_result_dtype();

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

template <>
at::Tensor LazyBinaryOp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = LazyOp<at::Tensor>::get_inputs();
  const auto& self = inputs.at(0).toTensor();

  const auto& outshape = LazyOp<at::Tensor>::get_out_shapes().empty()
      ? self.sizes()
      : LazyOp<at::Tensor>::get_out_shapes().at(0);

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
