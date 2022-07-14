/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/bitwise_left_shift.h"
#include "generated/bitwise_right_shift.h"
#include "hpu_op_helper.h"

namespace habana {

constexpr size_t index_of_self = 0;
constexpr size_t index_of_other = 1;

static std::shared_ptr<void> FillBitwiseShiftParams(
    const at::Stack&,
    ShiftDir_t shift_dir,
    size_t& size) {
  PARAMS_STUB(ns_BitShiftKernel::Params);
  params->direction = shift_dir;
  return params;
}

std::shared_ptr<void> FillLeftShiftParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBitwiseShiftParams(stack, ShiftDir_t::LEFT, size);
}

std::shared_ptr<void> FillRightShiftParams(
    const at::Stack& stack,
    size_t& size) {
  return FillBitwiseShiftParams(stack, ShiftDir_t::RIGHT, size);
}

static void ScalarTypeConvert(
    std::vector<at::IValue>& inputs,
    size_t scalar_index,
    size_t tensor_index) {
  auto tensor = inputs[tensor_index].toTensor();
  auto scalar = inputs[scalar_index].toScalar();
  if (c10::isFloatingType(tensor.scalar_type())) {
    inputs[scalar_index] = scalar.to<float>();
  } else {
    inputs[scalar_index] = scalar.to<int>();
  }
}

// To handle cases when scalar type !=tensor type
// type conversion similar to CPU implementation
// Jira raised for removing template specialization
// Jira link: https://jira.habana-labs.com/browse/SW-74866
template <>
ScalarTypeConversion<at::Tensor>::ScalarTypeConversion(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  auto x = get_inputs();
  if (x[index_of_self].isScalar()) {
    ScalarTypeConvert(x, index_of_self, index_of_other);
  } else {
    ScalarTypeConvert(x, index_of_other, index_of_self);
  }
  set_inputs(x);
}

template <>
at::Tensor ScalarTypeConversion<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  if (inputs.at(index_of_self).isScalar()) {
    const auto& t = inputs.at(index_of_other).toTensor();
    return habana_lazy::empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
  } else {
    const auto& t = inputs.at(index_of_self).toTensor();
    return habana_lazy::empty_hpu_lazy(
        t.sizes(), t.options(), t.suggest_memory_format(), false);
  }
}

} // namespace habana
