/******************************************************************************
 * Copyright (C) 2021-22 Habana Labs, Ltd. an Intel Company
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

#include "generated/clamp.h"
#include "generated/clamp_max.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {
// Use min/max of self tensor's dtype for clamping instead of
// blanket float limits. Use min/max of self's dtype as seen
// at backend(lowering).
// Clamp uses max/min guids which support only bf16, fp32 and int32.
// So the following table is limited to those dtypes.
// Long and Double at FE are seen as int and float at BE.
// Hence include these.
float self_type_max_for_be(c10::ScalarType type) {
  float max = std::numeric_limits<float>::max();
  switch (type) {
    case c10::ScalarType::Long:
    case c10::ScalarType::Int:
      // We should ideally use int max = 2147483647, But since
      // get_tensor_for_scalar() takes float value as argument,
      // we need to cast 2147483647 to float which becomes 2147483648.
      // This exceeds the int max limit. This causes issue down the line in
      // validateDownCast(). Hence use the largest integer value that
      // when converted to float becomes 2147483647. This is a tradeoff.
      // This int value is 2147483583, which is less than int max by 64.
      // Hence the clamping of these highest 64 integer values may not be
      // proper.
      // TODO:  Check if scalar caching can take types other than float in
      // the cache map. If yes, try using what ever is self's scalartype
      // instead of blanket float.
      max = (float)2147483583;
      break;

    case c10::ScalarType::Double:
    case c10::ScalarType::Float:
      max = (float)std::numeric_limits<float>::max();
      break;
    case c10::ScalarType::BFloat16:
      max = 3.38953139E38;
      break;
    default:
      // TODO: handle other dtypes
      PT_KERNEL_WARN("Using float max for unsupported type", type)
  }
  return max;
}

float self_type_min_for_be(c10::ScalarType type) {
  float min = std::numeric_limits<float>::lowest();
  switch (type) {
    case c10::ScalarType::Long:
    case c10::ScalarType::Int:
      min = (float)std::numeric_limits<int>::lowest();
      break;

    case c10::ScalarType::Double:
    case c10::ScalarType::Float:
      min = std::numeric_limits<float>::lowest();
      break;
    case c10::ScalarType::BFloat16:
      min = -3.38953139E38;
      break;
    default:
      // TODO: handle other dtypes
      PT_KERNEL_WARN("Using float min for unsupported type", type)
  }
  return min;
}

static void convert_params_to_tensors(
    at::Stack& inputs,
    at::ScalarType compute_dtype) {
  const auto& self = inputs[0].toTensor();
  float min = inputs[1].isScalar() ? inputs[1].toScalar().to<float>()
                                   : self_type_min_for_be(compute_dtype);
  float max = inputs[2].isScalar() ? inputs[2].toScalar().to<float>()
                                   : self_type_max_for_be(compute_dtype);
  inputs[1] = habana_lazy::get_tensor_for_scalar(
      min, self.options().dtype(compute_dtype));
  inputs[2] = habana_lazy::get_tensor_for_scalar(
      max, self.options().dtype(compute_dtype));
}

template <>
LazyClamp<at::Tensor>::LazyClamp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
LazyClamp<at::Tensor&>::LazyClamp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  convert_params_to_tensors(
      get_inputs(), inputs.at(0).toTensor().scalar_type());
}

template <>
at::Tensor LazyClamp<at::Tensor>::get_result_overrideable() {
  auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  convert_params_to_tensors(inputs, get_scalar_type());
  const auto& t = inputs.at(0).toTensor();
  return habana_lazy::empty_hpu_lazy(
      t.sizes(),
      t.options().dtype(get_scalar_type()),
      t.suggest_memory_format(),
      false);
}

template <>
at::Tensor& LazyClamp<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

template <typename ScalarType>
static std::shared_ptr<void> ClampParams(
    ScalarType min,
    ScalarType max,
    size_t& size) {
  PARAMS_STUB(ns_ClampKernel::Params);

  get<ScalarType>(params->lowerBound) = min;
  get<ScalarType>(params->upperBound) = max;

  return params;
}

std::shared_ptr<void> FillClampParams(const at::Stack& stack, size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    float min = stack[1].isScalar() ? stack[1].toScalar().to<float>()
                                    : -std::numeric_limits<float>::max();
    float max = stack[2].isScalar() ? stack[2].toScalar().to<float>()
                                    : std::numeric_limits<float>::max();
    return ClampParams(min, max, size);
  } else {
    int min = stack[1].isScalar() ? stack[1].toScalar().to<int>()
                                  : -std::numeric_limits<int>::max();
    int max = stack[2].isScalar() ? stack[2].toScalar().to<int>()
                                  : std::numeric_limits<int>::max();
    return ClampParams(min, max, size);
  }
}

std::shared_ptr<void> FillClampMinParams(const at::Stack& stack, size_t& size) {
  auto dtype_helper =
      habana_helpers::DTypeHelper::binary_op_with_type_promotion(
          stack, c10::nullopt, false);

  c10::ScalarType result_type = dtype_helper.get_result_dtype();

  if (c10::isFloatingType(result_type)) {
    return ClampParams(
        stack[1].toScalar().toFloat(), std::numeric_limits<float>::max(), size);
  }
  return ClampParams(
      stack[1].toScalar().toInt(), std::numeric_limits<int>::max(), size);
}

std::shared_ptr<void> FillClampMaxParams(const at::Stack& stack, size_t& size) {
  auto dtype_helper =
      habana_helpers::DTypeHelper::binary_op_with_type_promotion(
          stack, c10::nullopt, false);

  c10::ScalarType result_type = dtype_helper.get_result_dtype();

  if (c10::isFloatingType(result_type)) {
    return ClampParams(
        -std::numeric_limits<float>::max(),
        stack[1].toScalar().toFloat(),
        size);
  }
  return ClampParams(
      -std::numeric_limits<int>::max(), stack[1].toScalar().toInt(), size);
}

void clampTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  const auto& outshape = stack_tensor(stack, 0).sizes();
  bool minTensorDefined = stack.at(1).isTensor();
  bool maxTensorDefined = stack.at(2).isTensor();
  if (minTensorDefined && maxTensorDefined) {
    auto maxOut = BuildOp(
        graph,
        "max_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType()}});

    auto minOut = BuildOp(
        graph,
        "min_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {maxOut[0].get(), syn_in(2)},
        {{outshape, ScalarType(), 0}});

    syn_out(0) = std::move(minOut[0]);
  } else if (minTensorDefined) {
    auto maxOut = BuildOp(
        graph,
        "max_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType(), 0}});

    syn_out(0) = std::move(maxOut[0]);
  } else {
    HABANA_ASSERT(
        maxTensorDefined, "At least one of 'min' or 'max' must not be None")
    auto minOut = BuildOp(
        graph,
        "min_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType(), 0}});

    syn_out(0) = std::move(minOut[0]);
  }
}

} // namespace habana
