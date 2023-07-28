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
    std::optional<synTensor> syn_in0,
    std::optional<synTensor> syn_in1,
    std::string guid,
    std::vector<int64_t> outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_result_index) {
  std::vector<synTensor> inputs = {};
  if (can_use_dynamic_shapes(start, end, step)) {
    // It is assumend that syn_in1 and syn_in0 are non empty optionals when
    // can_use_dynamic_shapes returns ture
    inputs.emplace_back(syn_in1.value());
    inputs.emplace_back(syn_in0.value());
    const bool is_cast_not_required =
        habana_helpers::getInternalDtype(out_dtype) == c10::ScalarType::Int;
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

      return cast_to_out_type;
    }
  } else {
    op->CreateShapeTensorInput(graph, op->ScalarType(), outshape, inputs);
    auto internal_out_dtype = habana_helpers::getInternalDtype(out_dtype);
    const bool is_cast_not_required = c10::isFloatingType(internal_out_dtype) ||
        internal_out_dtype == c10::ScalarType::Int ||
        (internal_out_dtype == c10::ScalarType::Long &&
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
      return cast_to_out_type;
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
  std::optional<synTensor> syn_in0 = can_use_dynamic_shapes(start, end, step)
      ? std::make_optional(syn_in(0))
      : std::nullopt;
  std::optional<synTensor> syn_in1 = can_use_dynamic_shapes(start, end, step)
      ? std::make_optional(syn_in(1))
      : std::nullopt;
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

OutputMetaData ArangeDefaultCommonMeta(
    const int64_t depth,
    const at::IValue& dtype_opt,
    const at::IValue& layout_opt,
    const at::IValue& device_opt,
    const at::IValue& pin_memory_opt,
    const bool setToIntegralDType) {
  OutputMetaData meta;

  meta.dtype = dtype_opt.toOptional<at::ScalarType>().value_or(
      setToIntegralDType ? at::ScalarType::Long
                         : torch::get_default_dtype_as_scalartype());
  meta.layout = layout_opt.toOptional<at::Layout>().value_or(at::kStrided);
  auto device = device_opt.toOptional<at::Device>().value_or(at::kHPU);
  TORCH_INTERNAL_ASSERT(device.is_hpu());

  auto pin_memory = pin_memory_opt.toOptional<bool>().value_or(false);
  TORCH_CHECK(!pin_memory, "Only dense CPU tensors can be pinned");

  meta.shape = {depth};
  meta.mem_format = at::MemoryFormat::Contiguous;
  return {meta};
}

OutputMetaDataVector ArangeDefaultEndMeta(const at::Stack& stack) {
  const c10::Scalar defaultStart{0};
  const c10::Scalar defaultStep{1};
  const c10::Scalar end = stack.at(0).toScalar();
  const int64_t depth = get_arange_depth(defaultStart, end, defaultStep);
  const bool setToIntegralDType = end.isIntegral(true);
  return {ArangeDefaultCommonMeta(
      depth,
      stack.at(1),
      stack.at(2),
      stack.at(3),
      stack.at(4),
      setToIntegralDType)};
}

OutputMetaDataVector ArangeDefaultStartEndMeta(const at::Stack& stack) {
  const c10::Scalar start = stack.at(0).toScalar();
  const c10::Scalar defaultStep{1};
  const c10::Scalar end = stack.at(1).toScalar();
  const int64_t depth = get_arange_depth(start, end, defaultStep);
  const bool setToIntegralDType =
      end.isIntegral(true) && start.isIntegral(true);
  return {ArangeDefaultCommonMeta(
      depth,
      stack.at(2),
      stack.at(3),
      stack.at(4),
      stack.at(5),
      setToIntegralDType)};
}

OutputMetaDataVector ArangeDefaultStartEndStepMeta(const at::Stack& stack) {
  const c10::Scalar start = stack.at(0).toScalar();
  const c10::Scalar step = stack.at(2).toScalar();
  const c10::Scalar end = stack.at(1).toScalar();
  const int64_t depth = get_arange_depth(start, end, step);
  const bool setToIntegralDType =
      end.isIntegral(true) && start.isIntegral(true) && step.isIntegral(true);

  return {ArangeDefaultCommonMeta(
      depth,
      stack.at(3),
      stack.at(4),
      stack.at(5),
      stack.at(6),
      setToIntegralDType)};
}

std::shared_ptr<void> FillArangeDefaultCommonParams(
    c10::Scalar start,
    c10::Scalar end,
    c10::Scalar step,
    c10::ScalarType out_dtype,
    size_t& size) {
  auto internal_out_dtype = habana_helpers::getInternalDtype(out_dtype);
  PARAMS_STUB(ns_RangeKernel::Params);
  if (c10::isFloatingType(internal_out_dtype)) {
    params->start.f = start.to<float>();
    params->limit.f = end.to<float>();
    params->delta.f = step.to<float>();
  } else {
    params->start.i = start.to<int>();
    params->limit.i = end.to<int>();
    params->delta.i = step.to<int>();
  }
  return params;
}

std::shared_ptr<void> FillArangeDefaultEndParams(
    const at::Stack& stack,
    size_t& size) {
  const c10::Scalar defaultStart{0};
  const c10::Scalar defaultStep{1};
  const c10::Scalar end = stack.at(0).toScalar();
  const bool setToIntegralDType = end.isIntegral(true);

  const auto out_dtype = stack.at(1).toOptional<at::ScalarType>().value_or(
      setToIntegralDType ? at::ScalarType::Long
                         : torch::get_default_dtype_as_scalartype());
  return FillArangeDefaultCommonParams(
      defaultStart, end, defaultStep, out_dtype, size);
}

std::shared_ptr<void> FillArangeDefaultStartEndParams(
    const at::Stack& stack,
    size_t& size) {
  const c10::Scalar start = stack.at(0).toScalar();
  const c10::Scalar defaultStep{1};
  const c10::Scalar end = stack.at(1).toScalar();

  const bool setToIntegralDType =
      end.isIntegral(true) && start.isIntegral(true);

  const auto out_dtype = stack.at(2).toOptional<at::ScalarType>().value_or(
      setToIntegralDType ? at::ScalarType::Long
                         : torch::get_default_dtype_as_scalartype());

  return FillArangeDefaultCommonParams(
      start, end, defaultStep, out_dtype, size);
}

std::shared_ptr<void> FillArangeDefaultStartEndStepParams(
    const at::Stack& stack,
    size_t& size) {
  const c10::Scalar start = stack.at(0).toScalar();
  const c10::Scalar step = stack.at(2).toScalar();
  const c10::Scalar end = stack.at(1).toScalar();

  const bool setToIntegralDType =
      end.isIntegral(true) && start.isIntegral(true) && step.isIntegral(true);
  const auto out_dtype = stack.at(3).toOptional<at::ScalarType>().value_or(
      setToIntegralDType ? at::ScalarType::Long
                         : torch::get_default_dtype_as_scalartype());

  return FillArangeDefaultCommonParams(start, end, step, out_dtype, size);
}

synapse_helpers::tensor ArangeDefaultCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const OutputMetaDataVector& meta,
    std::shared_ptr<void> params,
    size_t params_size) {
  constexpr int FINAL_RESULT_INDEX = 0;
  const auto outshape = meta[0].shape;
  const auto out_dtype = meta[0].dtype;
  std::vector<synTensor> inputs = {};
  op->CreateShapeTensorInput(graph, op->ScalarType(), outshape, inputs);

  const auto internal_out_dtype = habana_helpers::getInternalDtype(out_dtype);
  const bool is_cast_not_required = c10::isFloatingType(internal_out_dtype) ||
      internal_out_dtype == c10::ScalarType::Int;
  auto scalar_type = is_cast_not_required ? out_dtype : c10::ScalarType::Int;
  auto range_guid = is_cast_not_required
      ? get_guid_with_precision("range", scalar_type)
      : "range_i32";
  NodeAttr::NodeOutputAttr out_attr = {outshape, scalar_type};
  if (is_cast_not_required)
    out_attr.final_result_index = FINAL_RESULT_INDEX;

  auto arange = OpBackend::BuildNode(
      op, graph, {range_guid, {inputs}, {out_attr}, params.get(), params_size});

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
        FINAL_RESULT_INDEX);
    return cast_to_out_type;
  }
}

void ArangeDefaultEnd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = OutputMeta(stack);

  size_t params_size = 0; // Will be set in FillArangeDefaultParams function
  auto params = FillParams(stack, params_size);
  syn_out(0) = ArangeDefaultCommon(this, graph, meta, params, params_size);
}

void ArangeDefaultStartEnd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = OutputMeta(stack);

  size_t params_size = 0; // Will be set in FillArangeDefaultParams function
  auto params = FillParams(stack, params_size);

  syn_out(0) = ArangeDefaultCommon(this, graph, meta, params, params_size);
}

void ArangeDefaultStartEndStep::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = OutputMeta(stack);

  size_t params_size = 0; // Will be set in FillArangeDefaultParams function
  auto params = FillParams(stack, params_size);

  syn_out(0) = ArangeDefaultCommon(this, graph, meta, params, params_size);
}

} // namespace habana
