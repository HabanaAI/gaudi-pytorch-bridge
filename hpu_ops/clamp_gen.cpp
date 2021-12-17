/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "hpu_op_helper.h"

namespace habana {
static void convert_params_to_tensors(std::vector<at::IValue>& inputs) {
  auto self = inputs[0].toTensor();
  // Scalar values always extracted as float irrespective of the type of
  // original value contained in Scalar. This is ok since the widest tensor type
  // which is supported on device is float so there is no chance of precision
  // loss.
  float min = inputs[1].isScalar() ? inputs[1].toScalar().to<float>()
                                   : -std::numeric_limits<float>::max();
  float max = inputs[2].isScalar() ? inputs[2].toScalar().to<float>()
                                   : std::numeric_limits<float>::max();
  auto min_tr = habana_lazy::get_tensor_for_scalar(min, self.options());
  auto max_tr = habana_lazy::get_tensor_for_scalar(max, self.options());
  inputs[1] = c10::IValue(min_tr);
  inputs[2] = c10::IValue(max_tr);
}

template <>
LazyClamp<at::Tensor>::LazyClamp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  auto x = get_inputs();
  convert_params_to_tensors(x);
  set_inputs(x);
}

template <>
LazyClamp<at::Tensor&>::LazyClamp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  auto x = get_inputs();
  convert_params_to_tensors(x);
  set_inputs(x);
}

template <>
at::Tensor LazyClamp<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  return habana_lazy::empty_hpu_lazy(
      t.sizes(), t.options(), t.suggest_memory_format(), false);
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
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    return ClampParams(
        stack[1].toScalar().toFloat(), std::numeric_limits<float>::max(), size);
  }
  return ClampParams(
      stack[1].toScalar().toInt(), std::numeric_limits<int>::max(), size);
}

std::shared_ptr<void> FillClampMaxParams(const at::Stack& stack, size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
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
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const at::Tensor self = stack_tensor(stack, 0);
  const auto& outshape = stack_tensor(stack, 0).sizes();
  bool minTensorDefined = stack.at(1).isTensor();
  bool maxTensorDefined = stack.at(2).isTensor();
  if (minTensorDefined && maxTensorDefined) {
    auto maxOut = BuildOp(
        graph,
        "max_fwd_" + habana_helpers::name_suffix_from_type(self.scalar_type()),
        {syn_in(0), syn_in(1)},
        {{outshape, self.scalar_type()}});

    auto minOut = BuildOp(
        graph,
        "min_fwd_" + habana_helpers::name_suffix_from_type(self.scalar_type()),
        {maxOut[0].get(), syn_in(2)},
        {{outshape, self.scalar_type(), is_output_persistent_list[0], 0}});

    syn_out(0) = std::move(minOut[0]);
  } else if (minTensorDefined) {
    auto maxOut = BuildOp(
        graph,
        "max_fwd_" + habana_helpers::name_suffix_from_type(self.scalar_type()),
        {syn_in(0), syn_in(1)},
        {{outshape, self.scalar_type(), is_output_persistent_list[0], 0}});

    syn_out(0) = std::move(maxOut[0]);
  } else {
    HABANA_ASSERT(
        maxTensorDefined, "At least one of 'min' or 'max' must not be None")
    auto minOut = BuildOp(
        graph,
        "min_fwd_" + habana_helpers::name_suffix_from_type(self.scalar_type()),
        {syn_in(0), syn_in(1)},
        {{outshape, self.scalar_type(), is_output_persistent_list[0], 0}});

    syn_out(0) = std::move(minOut[0]);
  }
}

} // namespace habana
