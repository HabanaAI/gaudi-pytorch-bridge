/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "common/utils.h"
#include "generated/backend/arange.h"
#include "hpu_ops/backend/arange.h"

namespace habana {

static bool can_use_dynamic_shapes(
    const c10::Scalar& start,
    const c10::Scalar& end,
    const c10::Scalar& step,
    const bool is_eager = false) {
  // Currently synapse support dynamic shape arange only for int datatypes.
  // For any other output datatype, will fallback to normal flow.
  return ((is_eager? habana_helpers::GetRefineDynamicShapeStatus(): true) &&
       habana_helpers::GetArangeHostTensorStatus() &&
      ((start.isIntegral(false) || can_convert(start)) &&
       (end.isIntegral(false) || can_convert(end)) &&
       (step.isIntegral(false) || can_convert(step))));
}

static int64_t get_arange_depth(
    const c10::Scalar _start,
    const c10::Scalar _end,
    const c10::Scalar _step) {
  const double start = _start.toDouble();
  const double end = _end.toDouble();
  const double step = _step.toDouble();

  TORCH_CHECK(step != 0.0, "step value can not be 0.");
  TORCH_CHECK(!((start > end) && (step > 0)), "step must be negative.");
  TORCH_CHECK(!((start < end) && (step < 0)), "step must be positive.");
  TORCH_CHECK(
      std::isfinite(start) && std::isfinite(end),
      "unsupported range: ",
      start,
      " -> ",
      end);

  double elements = std::ceil((end - start) / step);

  TORCH_CHECK(
      elements >= 0 &&
          elements <= static_cast<double>(std::numeric_limits<int64_t>::max()),
      "invalid number of elements, possible overflow");

  int64_t num_elements = static_cast<int64_t>(elements);
  return num_elements;
}

static int64_t get_arange_depth_ds(
    const float start,
    const float end,
    const float step) {
  TORCH_CHECK(step != 0.0, "step value can not be 0.");
  TORCH_CHECK(!((start > end) && (step > 0)), "step must be negative.");
  TORCH_CHECK(!((start < end) && (step < 0)), "step must be positive.");

  int64_t num_elements = static_cast<int64_t>(ceil((end - start) / step));
  return num_elements;
}

size_t GetMInMaxSifOffset(bool dry_run, size_t data_size) {
  size_t sif_offset = 0;
  if (dry_run &&
      habana::ShapeInference::GetCurrentPass() ==
          habana::ShapeInfo::InferencePass::MIN_SHAPE) {
    sif_offset = data_size;
  }
  return sif_offset;
}

template <typename T>
std::vector<T> GetArangeH2DParams(at::Tensor& params_t, bool dry_run) {
  std::vector<T> params_data;
  size_t data_size = params_t.sizes()[0];
  auto tmeta{get_tensor_extra_meta(params_t)};
  void* host_ptr = nullptr;
  if (dry_run) {
    host_ptr = tmeta->get_compile_host_ptr();
  } else {
    host_ptr = tmeta->get_host_ptr();
  }

  T* h2d_data = static_cast<T*>(host_ptr);
  size_t sif_offset = GetMInMaxSifOffset(dry_run, data_size);
  h2d_data = h2d_data + sif_offset;
  for (size_t i = 0; i < data_size; i++) {
    params_data.push_back(static_cast<T>(*h2d_data++));
  }
  return params_data;
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
  if (!c10::isFloatingType(out_scalar_type)) {
    // These parameters are used within GUID (range_i32).
    // If parameters are integer, start is rounded to floor while
    // limit (end) is rounded to ceiling.
    // This is to ensure the correct output size is calculated
    // from GUID (as start value is included in the output) and it
    // matches with the actual output size.
    params->start.i = static_cast<int>(floor(start.to<float>()));
    params->limit.i = static_cast<int>(ceil(end.to<float>()));
    params->delta.i = static_cast<int>(ceil(step.to<float>()));
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
    c10::optional<int> final_result_index,
    bool is_eager) {
  std::vector<synTensor> inputs = {};
  if (syn_in0.has_value() && can_use_dynamic_shapes(start, end, step, is_eager)) {
    // syn_in0 is defined, syn_in1 is optional
    // For arange.start_out, both syn_in0 and syn_in1 are defined.
    // For arange.start_step, only syn_in0 is defined.
    inputs.emplace_back(syn_in0.value());

    auto internal_out_dtype = habana_helpers::getInternalDtype(out_dtype);
    const bool is_cast_not_required = c10::isFloatingType(internal_out_dtype) ||
        internal_out_dtype == c10::ScalarType::Int ||
        (internal_out_dtype == c10::ScalarType::Long &&
         common::IsInt64Supported());
    auto scalar_type = is_cast_not_required ? out_dtype : c10::ScalarType::Int;
    auto range_guid = get_guid_with_precision("range", scalar_type);
    NodeAttr::NodeOutputAttr out_attr = {outshape, scalar_type};

    if (is_cast_not_required)
      out_attr.final_result_index = final_result_index;

    std::vector<synapse_helpers::tensor> arange_i32 {};

    if (!syn_in1.has_value()) {
       // arange.start_step
       // Only syn_in0 is defined
       arange_i32 = OpBackend::BuildNode(
          op, graph, {range_guid, std::move(inputs), {out_attr}});
    }
    else {
       // arange.start_out
       // syn_in1 is defined
       inputs.emplace_back(syn_in1.value());
       op->CreateShapeTensorInput(graph, op->ScalarType(), outshape, inputs);
       arange_i32 = OpBackend::BuildNode(
          op, graph, {range_guid, {}, {out_attr}, params.get(), size});
    }

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
         common::IsInt64Supported());
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

OutputMetaDataVector ArangeDefaultCommonMeta(
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

OutputMetaDataVector ArangeStartOutMeta(const at::Stack& stack) {
  const c10::Scalar start = stack.at(0).toScalar();
  const c10::Scalar step = stack.at(2).toScalar();
  const c10::Scalar end = stack.at(1).toScalar();
  const auto outScalarType = stack.back().toTensor().scalar_type();
  const int64_t depth = get_arange_depth(start, end, step);
  const bool setToIntegralDType =
      end.isIntegral(true) && start.isIntegral(true) && step.isIntegral(true);

  return ArangeDefaultCommonMeta(
      depth,
      outScalarType,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      setToIntegralDType);
}

void Arange::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  bool is_eager = GetExecutionMode() != habana_helpers::HabanaFrontendTypes::COMPILE;
  const auto meta = ArangeStartOutMeta(stack)[0];
  auto outshape = meta.shape;
  auto out_dtype = meta.dtype;
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto start = stack.at(0).toScalar();
  auto end = stack.at(1).toScalar();
  auto step = stack.at(2).toScalar();
  std::optional<synTensor> syn_in0 = (p_context_->syn_inputs_.size())
      ? std::make_optional(syn_in(0))
      : std::nullopt;
  std::optional<synTensor> syn_in1 = (p_context_->syn_inputs_.size())
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
      0,
      is_eager);
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
  int64_t depth;
  bool setToIntegralDType = true; // DS Compile

  if (!stack[0].isTensor()) {
    const c10::Scalar start = stack.at(0).toScalar();
    const c10::Scalar step = stack.at(2).toScalar();
    const c10::Scalar end = stack.at(1).toScalar();
    depth = get_arange_depth(start, end, step);
    const bool setToIntegralDType =
        end.isIntegral(true) && start.isIntegral(true) && step.isIntegral(true);

    return {ArangeDefaultCommonMeta(
        depth,
        stack.at(3),
        stack.at(4),
        stack.at(5),
        stack.at(6),
        setToIntegralDType)};
  } else {
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1 ||
        GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
      // Lazy Flow
      auto output_shape_tensor = stack[1].toTensor();
      depth = output_shape_tensor.sizes()[0];
      setToIntegralDType =
          (output_shape_tensor.scalar_type() == c10::ScalarType::Long) ||
          (output_shape_tensor.scalar_type() == c10::ScalarType::Int);
      return {ArangeDefaultCommonMeta(
          depth,
          stack.at(2),
          stack.at(3),
          stack.at(4),
          stack.at(5),
          setToIntegralDType)};
    } else {
      // DS Compile Flow
      std::vector<int32_t> params_data;
      at::Tensor params_t = stack[0].toTensor();
      if ((habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MIN_SHAPE) ||
          (habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
        params_data = GetArangeH2DParams<int32_t>(params_t, true);
      } else {
        params_data = GetArangeH2DParams<int32_t>(params_t, false);
      }

      PT_KERNEL_DEBUG("ArangeOutputShape params_data:", params_data);
      depth = get_arange_depth_ds(
          static_cast<float>(params_data[0]),
          static_cast<float>(params_data[1]),
          static_cast<float>(params_data[2]));

      return {ArangeDefaultCommonMeta(
          depth,
          stack.at(2),
          stack.at(3),
          stack.at(4),
          stack.at(5),
          setToIntegralDType)};
    }
  }
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
    // These parameters are used within GUID (range_i32).
    // If parameters are integer, start is rounded to floor while
    // limit (end) is rounded to ceiling.
    // This is to ensure the correct output size is calculated
    // from GUID (as start value is included in the output) and it
    // matches with the actual output size.
    params->start.i = static_cast<int>(floor(start.to<float>()));
    params->limit.i = static_cast<int>(ceil(end.to<float>()));
    params->delta.i = static_cast<int>(ceil(step.to<float>()));
  }
  return params;
}

std::shared_ptr<void> FillArangeDefaultCommonParamsDS(
    const int32_t start,
    const int32_t end,
    const int32_t step,
    // c10::ScalarType out_dtype,
    size_t& size) {
  // auto internal_out_dtype = habana_helpers::getInternalDtype(out_dtype);
  PARAMS_STUB(ns_RangeKernel::Params);
  // Input parameters are integers. No casting/rounding needed.
  params->start.i = start;
  params->limit.i = end;
  params->delta.i = step;
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
  if (!stack[0].isTensor()) {
    const c10::Scalar start = stack.at(0).toScalar();
    const c10::Scalar step = stack.at(2).toScalar();
    const c10::Scalar end = stack.at(1).toScalar();

    const bool setToIntegralDType =
        end.isIntegral(true) && start.isIntegral(true) && step.isIntegral(true);
    const auto out_dtype = stack.at(3).toOptional<at::ScalarType>().value_or(
        setToIntegralDType ? at::ScalarType::Long
                           : torch::get_default_dtype_as_scalartype());

    return FillArangeDefaultCommonParams(start, end, step, out_dtype, size);
  } else {
    const auto meta = ArangeDefaultStartEndStepMeta(stack);
    const auto out_dtype = meta[0].dtype;
    if (c10::isFloatingType(out_dtype)) {
      std::vector<float> params_data;
      at::Tensor params_t = stack[0].toTensor();
      if ((habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MIN_SHAPE) ||
          (habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
        params_data = GetArangeH2DParams<float>(params_t, true);
      } else {
        params_data = GetArangeH2DParams<float>(params_t, false);
      }
      return FillArangeDefaultCommonParams(
          params_data[0], params_data[1], params_data[2], out_dtype, size);
    } else {
      std::vector<int32_t> params_data;
      at::Tensor params_t = stack[0].toTensor();
      if ((habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MIN_SHAPE) ||
          (habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
        params_data = GetArangeH2DParams<int32_t>(params_t, true);
      } else {
        params_data = GetArangeH2DParams<int32_t>(params_t, false);
      }
      return FillArangeDefaultCommonParams(
          params_data[0], params_data[1], params_data[2], out_dtype, size);
    }
  }
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

static SharedMetaDataVector ArangeDefaultSharedMeta(
    const bool is_integral,
    const at::optional<at::ScalarType>& dtype) {
  const auto out_dtype = dtype.value_or(
      is_integral ? at::ScalarType::Long
                  : torch::get_default_dtype_as_scalartype());
  const auto internal_out_dtype = habana_helpers::getInternalDtype(out_dtype);
  const bool is_cast_not_required = c10::isFloatingType(internal_out_dtype) ||
      internal_out_dtype == c10::ScalarType::Int;
  const auto range_type =
      is_cast_not_required ? out_dtype : c10::ScalarType::Int;

  SharedMetaData range{"range"};
  range.outputs_data = {{1, range_type}};

  return {range};
}

SharedMetaDataVector ArangeDefaultEndSharedMeta(const at::Stack& stack) {
  const auto end = stack[0].toScalar();
  return ArangeDefaultSharedMeta(
      end.isIntegral(true), stack[1].toOptional<at::ScalarType>());
}

SharedMetaDataVector ArangeDefaultStartEndSharedMeta(const at::Stack& stack) {
  const auto start = stack[0].toScalar();
  const auto end = stack[1].toScalar();
  return ArangeDefaultSharedMeta(
      start.isIntegral(true) and end.isIntegral(true),
      stack[2].toOptional<at::ScalarType>());
}

SharedMetaDataVector ArangeDefaultStartStepSharedMeta(const at::Stack& stack) {
  const auto start = stack[0].toScalar();
  const auto end = stack[1].toScalar();
  const auto step = stack[2].toScalar();
  return ArangeDefaultSharedMeta(
      start.isIntegral(true) and end.isIntegral(true) and step.isIntegral(true),
      stack[3].toOptional<at::ScalarType>());
}

SharedMetaDataVector ArangeDefaultStartOutSharedMeta(const at::Stack& stack) {
  const auto start = stack[0].toScalar();
  const auto end = stack[1].toScalar();
  const auto step = stack[2].toScalar();
  return ArangeDefaultSharedMeta(
      start.isIntegral(true) and end.isIntegral(true) and step.isIntegral(true),
      stack[3].toTensor().scalar_type());
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
  bool is_eager = GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER;
  const auto meta = OutputMeta(stack);
  const auto outshape = meta[0].shape;
  const auto out_dtype = meta[0].dtype;
  size_t params_size = 0; // Will be set in FillArangeDefaultParams function
  auto params = FillParams(stack, params_size);

  const auto internal_out_dtype = habana_helpers::getInternalDtype(out_dtype);

  c10::Scalar start, end, step;
  if (!stack[0].isTensor()) {
    start = stack.at(0).toScalar();
    end = stack.at(1).toScalar();
    step = stack.at(2).toScalar();
    syn_out(0) = ArangeDefaultCommon(this, graph, meta, params, params_size);
  } else {
    if (c10::isFloatingType(internal_out_dtype)) {
      std::vector<float> params_data;
      at::Tensor params_t = stack[0].toTensor();
      if ((habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MIN_SHAPE) ||
          (habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
        params_data = GetArangeH2DParams<float>(params_t, true);
      } else {
        params_data = GetArangeH2DParams<float>(params_t, false);
      }
      start = params_data[0];
      end = params_data[1];
      step = params_data[2];
    } else {
      std::vector<int32_t> params_data;
      at::Tensor params_t = stack[0].toTensor();
      if ((habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MIN_SHAPE) ||
          (habana::ShapeInference::GetCurrentPass() ==
           habana::ShapeInfo::InferencePass::MAX_SHAPE)) {
        params_data = GetArangeH2DParams<int32_t>(params_t, true);
      } else {
        params_data = GetArangeH2DParams<int32_t>(params_t, false);
      }
      start = params_data[0];
      end = params_data[1];
      step = params_data[2];
    }

    std::optional<synTensor> syn_in0 = syn_in(0);
    std::optional<synTensor> syn_in1 = std::nullopt;

    if (p_context_->syn_inputs_.size() == 2) {
      EraseSynInput(1);
    }

    syn_out(0) = ArangeCommon(
        this,
        graph,
        start,
        end,
        step,
        internal_out_dtype,
        syn_in0,
        syn_in1,
        guid_,
        outshape,
        params,
        params_size,
        0,
        is_eager);
  }
}

} // namespace habana
