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
        {{outshape, self.scalar_type(), false}});

    auto minOut = BuildOp(
        graph,
        "min_fwd_" + habana_helpers::name_suffix_from_type(self.scalar_type()),
        {maxOut[0].get(), syn_in(2)},
        {{outshape, self.scalar_type(), is_output_persistent_list[0], true}});

    syn_out(0) = std::move(minOut[0]);
  } else if (minTensorDefined) {
    auto maxOut = BuildOp(
        graph,
        "max_fwd_" + habana_helpers::name_suffix_from_type(self.scalar_type()),
        {syn_in(0), syn_in(1)},
        {{outshape, self.scalar_type(), is_output_persistent_list[0], true}});

    syn_out(0) = std::move(maxOut[0]);
  } else {
    HABANA_ASSERT(
        maxTensorDefined, "At least one of 'min' or 'max' must not be None")
    auto minOut = BuildOp(
        graph,
        "min_fwd_" + habana_helpers::name_suffix_from_type(self.scalar_type()),
        {syn_in(0), syn_in(1)},
        {{outshape, self.scalar_type(), is_output_persistent_list[0], true}});

    syn_out(0) = std::move(minOut[0]);
  }
}

} // namespace habana
