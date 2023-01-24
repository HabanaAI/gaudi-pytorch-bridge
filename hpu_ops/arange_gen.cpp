/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/arange.h"
#include "hpu_op_helper.h"

namespace habana {

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

static bool can_convert(const c10::Scalar& value) {
  if (value.isFloatingPoint()) {
    auto float_value = value.toFloat();
    auto int_value = value.toInt();
    auto diff = float_value - int_value;
    return !(diff > 0);
  }
  return true;
}

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

template <>
ArangeInputs<at::Tensor&>::ArangeInputs(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  auto start = inputs[0].toScalar();
  auto end = inputs[1].toScalar();
  auto step = inputs[2].toScalar();
  auto output = inputs[3].toTensor();

  if (can_use_dynamic_shapes(start, end, step)) {
    std::vector<int32_t> params_vec{start.toInt(), end.toInt(), step.toInt()};
    auto params_shape = habana_lazy::empty_hpu_lazy(
        params_vec.size(),
        output.options(),
        output.suggest_memory_format(),
        false,
        HOST_TO_DEVICE_TENSOR);
    auto hl_params_shape =
        habana_lazy::GetOrCreateHbLazyTensor(params_shape, c10::kHPU);

    auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
    habana_lazy::HbInternalTensorImpl* impl =
        habana_lazy::GetHbInternalTensorImpl(hl_param_internal);
    HABANA_ASSERT(impl);
    impl->set_host_data(
        params_vec.data(),
        params_vec.size(),
        sizeof(int),
        habana_lazy::HostDataType::INT32_T);

    // Create a dummy shape tensor for the output, this shape tensor is not
    // added to synapse graph, but only ensures that when we match in bucket
    // we are restricted by the size of the output
    int out_depth = get_arange_depth(start, end, step);
    auto out_shape = c10::DimVector({out_depth});
    auto result_shape = habana_lazy::empty_hpu_lazy(
        out_shape,
        output.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);
    // Mark this front end shape tensor as it does not need synapse tensor
    auto hl_result_shape =
        habana_lazy::GetOrCreateHbLazyTensor(result_shape, c10::kHPU);
    auto hl_result_shape_internal =
        hl_result_shape.CurrentTensorAttached().value();
    auto stImpl =
        habana_lazy::GetHbInternalTensorImpl(hl_result_shape_internal);
    if (stImpl) {
      stImpl->setH2DFrontEndShapeTensor();
    }
    set_inputs({start, end, step, params_shape, result_shape, inputs[3]});
  } else {
    set_inputs({start, end, step, {}, {}, inputs[3]});
  }
}

template <>
at::Tensor& ArangeInputs<at::Tensor&>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

sizes_vec ArangeOutputShape(const at::Stack& stack) {
  return {{get_arange_depth(
      stack.at(0).toScalar(), stack.at(1).toScalar(), stack.at(2).toScalar())}};
}

std::shared_ptr<void> FillArangeParams(const at::Stack& stack, size_t& size) {
  const c10::Scalar start = stack.at(0).toScalar();
  const c10::Scalar end = stack.at(1).toScalar();
  const c10::Scalar step = stack.at(2).toScalar();
  auto out_tensor = stack.back().toTensor();

  PARAMS_STUB(ns_RangeKernel::Params);
  if (can_use_dynamic_shapes(start, end, step) ||
      !c10::isFloatingType(out_tensor.scalar_type())) {
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

void Arange::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = ComputeOutputShapes(stack)[0];
  size_t size = 0;
  auto params = FillParams(stack, size);

  auto start = stack.at(0).toScalar();
  auto end = stack.at(1).toScalar();
  auto step = stack.at(2).toScalar();

  auto out_tensor = stack.back().toTensor();

  std::vector<synTensor> inputs = {};
  if (can_use_dynamic_shapes(start, end, step)) {
    inputs.emplace_back(syn_in(1));
    inputs.emplace_back(syn_in(0));
    const bool is_cast_not_required =
        out_tensor.scalar_type() == c10::ScalarType::Int;
    NodeAttr::NodeOutputAttr out_attr = {outshape, c10::ScalarType::Int};
    if (is_cast_not_required)
      out_attr.final_result_index = 0;
    auto arange_i32 = BuildOp(graph, "range_i32", inputs, {out_attr});
    if (is_cast_not_required) {
      syn_out(0) = std::move(arange_i32[0]);
    } else {
      auto cast_to_out_type = CastHelper(
          graph,
          arange_i32.at(0).get(),
          outshape,
          c10::ScalarType::Int,
          out_tensor.scalar_type(),
          0);
      syn_out(0) = std::move(cast_to_out_type);
    }
  } else {
    this->CreateShapeTensorInput(graph, this->ScalarType(), outshape, inputs);
    const bool is_cast_not_required =
        c10::isFloatingType(out_tensor.scalar_type()) ||
        out_tensor.scalar_type() == c10::ScalarType::Int ||
        (out_tensor.scalar_type() == c10::ScalarType::Long &&
         GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT));
    auto scalar_type =
        is_cast_not_required ? out_tensor.scalar_type() : c10::ScalarType::Int;
    auto range_guid = is_cast_not_required ? guid_ : "range_i32";
    NodeAttr::NodeOutputAttr out_attr = {outshape, scalar_type};
    if (is_cast_not_required)
      out_attr.final_result_index = 0;
    auto arange =
        BuildOp(graph, range_guid, {}, {out_attr}, params.get(), size);

    if (is_cast_not_required) {
      syn_out(0) = std::move(arange[0]);
    } else {
      auto cast_to_out_type = CastHelper(
          graph,
          arange.at(0).get(),
          outshape,
          c10::ScalarType::Int,
          out_tensor.scalar_type(),
          0);
      syn_out(0) = std::move(cast_to_out_type);
    }
  }
}
} // namespace habana