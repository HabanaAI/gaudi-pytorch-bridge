/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "div_mod_util.h"
#include "generated/div.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/binary_kernels.h"

// For use in div_rounding_mode
#define StrModeFloor "floor"
#define StrModeTruncate "trunc"

namespace habana {

static c10::ScalarType GetResultDtype(
    const std::vector<at::IValue>& inputs,
    bool int_to_float) {
  return habana_helpers::DTypeHelper::
      binary_op_with_optional_int_to_float_promotion(
             inputs, int_to_float, c10::nullopt, false)
          .get_result_dtype();
}

static c10::ScalarType GetCommonDtype(
    const std::vector<at::IValue>& inputs,
    bool int_to_float) {
  return habana_helpers::DTypeHelper::
      binary_op_with_optional_int_to_float_promotion(
             inputs, int_to_float, c10::nullopt, false)
          .get_common_dtype();
}

static bool DivCommonCheck(
    const at::Tensor& self,
    const c10::IValue& other,
    c10::optional<c10::string_view>&& rounding_mode) {
  auto promote_int_to_float = !rounding_mode;
  auto result_type = GetCommonDtype({self, other}, promote_int_to_float);
  if (!promote_int_to_float && c10::isIntegralType(result_type, true)) {
    return true;
  }
  switch (result_type) {
    case torch::kBFloat16:
    case torch::kFloat32:
    case torch::kFloat64:
      return true;
    case torch::kHalf: {
      auto device_type{synapse_helpers::HPURegistrar::get_device().type()};
      return device_type == synDeviceGaudi2 || device_type == synDeviceGreco ||
          device_type == synDeviceGaudi3;
    }
    case torch::kInt8:
    case torch::kInt16:
    case torch::kInt32:
    case torch::kInt64:
      // floor and trunc support integral types by casts
      return rounding_mode.has_value();
    default:
      return false;
  }
}

FALLBACK_CHECK(
    DivTensorModeFallbackCheck,
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  return DivCommonCheck(self, other, std::move(rounding_mode));
}

FALLBACK_CHECK(
    DivScalarModeFallbackCheck,
    const at::Tensor& self,
    const at::Scalar& other,
    c10::optional<c10::string_view> rounding_mode) {
  return DivCommonCheck(self, other, std::move(rounding_mode));
}

static void convert_scalar_to_tensor(
    at::Stack& stack,
    c10::optional<c10::ScalarType> compute_dtype = c10::nullopt) {
  auto& other_ival = stack.at(1);
  const auto& other = other_ival.toScalar();
  other_ival = habana_lazy::get_tensor_for_scalar(
      other.to<double>(), compute_dtype.value_or(other.type()));
}

template <>
LazyDivScalar<at::Tensor>::LazyDivScalar(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyDivScalar<at::Tensor>::get_result_overrideable() {
  auto& inputs = LazyOp<at::Tensor>::get_inputs();
  at::ScalarType result_dtype = get_scalar_type();
  convert_scalar_to_tensor(inputs, result_dtype);

  const auto& self = inputs[0].toTensor();
  return habana_lazy::empty_hpu_lazy(
      self.sizes(),
      self.options().dtype(result_dtype),
      self.suggest_memory_format(),
      false);
}

template <>
LazyDivScalarInplace<at::Tensor&>::LazyDivScalarInplace(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  convert_scalar_to_tensor(get_inputs());
}
template <>
at::Tensor& LazyDivScalarInplace<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

template <typename T>
static void div_mode(habana_lazy::LazyOp<T>* op, at::Stack& inputs) {
  c10::optional<c10::string_view> rounding_mode =
      inputs.at(2).toOptional<c10::string_view>();
  TORCH_CHECK(
      !rounding_mode.has_value() or (*rounding_mode == "trunc") or
          (*rounding_mode == StrModeFloor),
      "div expected rounding_mode to be one of None, '",
      StrModeTruncate,
      "', or '",
      StrModeFloor,
      "' "
      "but found '",
      *rounding_mode,
      "'");
  at::ScalarType result_type =
      GetResultDtype(inputs, !rounding_mode.has_value());
  op->set_scalar_type(result_type);
  if (inputs.at(1).isScalar()) {
    convert_scalar_to_tensor(inputs, result_type);
  }
}

template <>
DivMode<at::Tensor>::DivMode(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  div_mode(this, get_inputs());
}

template <>
at::Tensor DivMode<at::Tensor>::get_result_overrideable() {
  return LazyOp<at::Tensor>::get_result_overrideable();
}

template <>
DivMode<at::Tensor&>::DivMode(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  div_mode(this, get_inputs());
}

template <>
at::Tensor& DivMode<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

static std::vector<synapse_helpers::tensor> CommonFuncForRoundingModeIntType(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> inputs,
    std::vector<int64_t> shape_out,
    const c10::optional<c10::string_view>& rounding_mode,
    c10::ScalarType final_result_type) {
  std::vector<synapse_helpers::tensor> output;
  // The second argument of "FillDivModParams", pyCompatible is false
  // for 'trunc' mode and true for 'floor' case
  output = GetDivModOutput(
      op,
      graph,
      inputs[0],
      inputs[1],
      (StrModeFloor == rounding_mode),
      std::move(shape_out),
      final_result_type,
      DIV_MODE_OUTPUT_TYPE::QUOTIENT);
  return output;
}

std::vector<synapse_helpers::tensor> DivCommonFunction(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<synTensor> binaryop_inputs,
    const c10::optional<c10::string_view>& rounding_mode) {
  const at::Tensor& self = stack_tensor(stack, 0);
  const at::Tensor& other = stack_tensor(stack, 1);
  std::vector<at::Tensor> tensors = {self, other};

  // Check if mode is other than None, i.e. "floor" or "trunc"
  bool isNotNone = rounding_mode.has_value();

  // Find the result type
  auto final_result_type = GetResultDtype(stack, !isNotNone);

  auto shape_out = BinaryOperator::compute_output_shape(self, other);
  // Handle integral cases differently using div_mod, else floating point
  // convertion yields error after truncation in some cases.
  if (isNotNone && (c10::isIntegralType(final_result_type, true))) {
    auto res = CommonFuncForRoundingModeIntType(
        op,
        graph,
        binaryop_inputs,
        shape_out,
        rounding_mode,
        final_result_type);
    return res;
  } else { // if (isIntegralType(final_result_type, true))

    // Computation is always done in float or bfloat16
    at::ScalarType computation_type = GetCommonDtype(stack, !isNotNone);
    const std::string opStringSuffix =
        "_fwd_" + habana_helpers::name_suffix_from_type(computation_type);

    // Initialization
    const unsigned cNoOfInputTensors = 2;
    std::vector<synapse_helpers::tensor> divOp, makeIntegerOp;
    std::unique_ptr<synapse_helpers::tensor> cast[cNoOfInputTensors];

    // Convert each tensor to float/bfloat16 (if not already in)
    std::string strNode_type;

    for (unsigned char i = 0; i < cNoOfInputTensors; ++i) {
      if (tensors.at(i).scalar_type() == computation_type) {
        continue;
      }

      cast[i] = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
          op,
          graph,
          binaryop_inputs[i],
          stack_tensor(stack, i).sizes(),
          tensors.at(i).scalar_type(),
          computation_type));
      binaryop_inputs.at(i) = cast[i]->get();
    }

    // Final cast is required, if result type is not float
    // when those are same, and following flag bNeedToCastFinalResult is false
    // we used computation_type for output type to avoid multiple branches
    bool bNeedToCastFinalResult = (final_result_type != computation_type);

    auto guid = op->GetGuid();
    guid = guid.substr(0, guid.find_last_of('_') + 1) +
        habana_helpers::name_suffix_from_type(computation_type);
    divOp = OpBackend::BuildNode(
        op,
        graph,
        {guid,
         binaryop_inputs,
         {{shape_out,
           computation_type,
           isNotNone ? c10::nullopt : c10::make_optional<int>(0)}}});
    if (!isNotNone) {
      // when flow reaches here, computation_type is same as final_result_type,
      // so computation_type can be used as div's return type and that is the
      // return type of div_rounding_mode
      return divOp;
    }

    // If in "floor" or "trunc" mode, need to apply that
    makeIntegerOp = OpBackend::BuildNode(
        op,
        graph,
        {std::string(*rounding_mode) + opStringSuffix,
         {divOp.at(0).get()},
         {{shape_out,
           computation_type,
           bNeedToCastFinalResult ? c10::nullopt
                                  : c10::make_optional<int>(0)}}});
    if (!bNeedToCastFinalResult) {
      // when flow reaches here, computation_type is same as final_result_type,
      // so computation_type can be used as return type of floor/trunc and that
      // is the return type of div_rounding_mode
      return makeIntegerOp;
    }
    std::vector<synapse_helpers::tensor> castToReturnTypeOp;
    castToReturnTypeOp.push_back(OpBackend::BuildCast(
        op,
        graph,
        makeIntegerOp.at(0).get(),
        shape_out,
        computation_type,
        final_result_type,
        0));
    return castToReturnTypeOp;
  }
}

void DivRoundModeOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  c10::optional<c10::string_view> rounding_mode =
      stack.at(2).toOptional<c10::string_view>();
  std::vector<synTensor> binaryop_inputs{syn_in(0), syn_in(1)};
  auto out =
      DivCommonFunction(this, graph, stack, binaryop_inputs, rounding_mode);
  syn_out(0) = std::move(out[0]);
}
} // namespace habana
