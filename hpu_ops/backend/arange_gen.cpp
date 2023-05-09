/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/common/arange_gen.h"
#include "generated/backend/arange.h"
#include "hpu_ops/backend/arange.h"

namespace habana {

static bool can_use_dynamic_shapes(
    const c10::Scalar& start,
    const c10::Scalar& end,
    const c10::Scalar& step) {
  // Currently synapse support dynamic shape arange only for int datatypes.
  // For any other output datatype, will fallback to normal flow.
  return (
      (habana_helpers::GetRefineDynamicShapeStatus() &&
       GET_ENV_FLAG_NEW(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR)) &&
      ((start.isIntegral(false) || can_convert(start)) &&
       (end.isIntegral(false) || can_convert(end)) &&
       (step.isIntegral(false) || can_convert(step))));
}

static int64_t get_arange_depth(
    const c10::Scalar _start,
    const c10::Scalar _end,
    const c10::Scalar _step) {
  const float start = _start.to<float>();
  const float end = _end.to<float>();
  const float step = _step.to<float>();

  TORCH_CHECK(step != 0.0, "step value can not be 0.");
  TORCH_CHECK(!((start > end) && (step > 0)), "step must be negative.");
  TORCH_CHECK(!((start < end) && (step < 0)), "step must be positive.");

  int64_t num_elements = static_cast<int64_t>(ceil((end - start) / step));
  return num_elements;
}

sizes_vec ArangeOutputShape(const at::Stack& stack) {
  return {{get_arange_depth(
      stack.at(0).toScalar(), stack.at(1).toScalar(), stack.at(2).toScalar())}};
}

std::shared_ptr<void> FillArangeParams(const at::Stack& stack, size_t& size) {
  const c10::Scalar start = stack.at(0).toScalar();
  const c10::Scalar end = stack.at(1).toScalar();
  const c10::Scalar step = stack.at(2).toScalar();
  auto out_scalar_type = stack.back().toTensor().scalar_type();
  return FillArangeParamsInternal(start, end, step, out_scalar_type, size);
}

std::shared_ptr<void> FillArangeParamsInternal(
    c10::Scalar start,
    c10::Scalar end,
    c10::Scalar step,
    c10::ScalarType out_scalar_type,
    size_t& size) {
  PARAMS_STUB(ns_RangeKernel::Params);
  if (can_use_dynamic_shapes(start, end, step) ||
      !c10::isFloatingType(out_scalar_type)) {
    params->start.i = start.to<int>();
    params->limit.i = end.to<int>();
    params->delta.i = step.to<int>();
  } else {
    params->start.f = start.to<float>();
    params->limit.f = end.to<float>();
    params->delta.f = step.to<float>();
  }
  return params;
}

synapse_helpers::tensor ArangeCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    c10::Scalar start,
    c10::Scalar end,
    c10::Scalar step,
    c10::ScalarType out_dtype,
    synTensor syn_in0,
    synTensor syn_in1,
    std::string guid,
    std::vector<int64_t> outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_result_index) {
  std::vector<synTensor> inputs = {};
  if (can_use_dynamic_shapes(start, end, step)) {
    inputs.emplace_back(syn_in1);
    inputs.emplace_back(syn_in0);
    const bool is_cast_not_required = out_dtype == c10::ScalarType::Int;
    NodeAttr::NodeOutputAttr out_attr = {outshape, c10::ScalarType::Int};
    if (is_cast_not_required)
      out_attr.final_result_index = 0;

    auto arange_i32 = OpBackend::BuildNode(
        op, graph, {"range_i32", std::move(inputs), {out_attr}});
    if (is_cast_not_required) {
      return std::move(arange_i32[0]);
    } else {
      auto cast_to_out_type = OpBackend::BuildCast(
          op,
          graph,
          arange_i32.at(0).get(),
          outshape,
          c10::ScalarType::Int,
          out_dtype,
          final_result_index);

      return std::move(cast_to_out_type);
    }
  } else {
    op->CreateShapeTensorInput(graph, op->ScalarType(), outshape, inputs);
    const bool is_cast_not_required = c10::isFloatingType(out_dtype) ||
        out_dtype == c10::ScalarType::Int ||
        (out_dtype == c10::ScalarType::Long &&
         GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT));
    auto scalar_type = is_cast_not_required ? out_dtype : c10::ScalarType::Int;
    auto range_guid = is_cast_not_required ? guid : "range_i32";
    NodeAttr::NodeOutputAttr out_attr = {outshape, scalar_type};
    if (is_cast_not_required)
      out_attr.final_result_index = final_result_index;
    auto arange = OpBackend::BuildNode(
        op, graph, {range_guid, {}, {out_attr}, params.get(), size});

    if (is_cast_not_required) {
      return std::move(arange[0]);
    } else {
      auto cast_to_out_type = OpBackend::BuildCast(
          op,
          graph,
          arange.at(0).get(),
          outshape,
          c10::ScalarType::Int,
          out_dtype,
          final_result_index);
      return std::move(cast_to_out_type);
    }
  }
}

void Arange::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = ComputeOutputShapes(stack)[0];
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto start = stack.at(0).toScalar();
  auto end = stack.at(1).toScalar();
  auto step = stack.at(2).toScalar();
  auto out_dtype = stack.back().toTensor().scalar_type();
  synTensor s0, s1;
  synTensor syn_in0 =
      (can_use_dynamic_shapes(start, end, step)) ? syn_in(0) : s0;
  synTensor syn_in1 =
      (can_use_dynamic_shapes(start, end, step)) ? syn_in(1) : s1;
  syn_out(0) = ArangeCommon(
      this,
      graph,
      start,
      end,
      step,
      out_dtype,
      syn_in0,
      syn_in1,
      guid_,
      outshape,
      params,
      size,
      0);
}
} // namespace habana
