/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "../habana_kernels/lazy_kernels_declarations.h"
#include "div_mod_util.h"
#include "generated/div.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/binary_kernels.h"

// For use in div_rounding_mode
#define StrModeTrue ""
#define StrModeFloor "floor"
#define StrModeTruncate "trunc"

namespace habana {

static bool DivCommonCheck(
    const at::Tensor& self,
    const c10::IValue& other,
    c10::optional<c10::string_view> rounding_mode) {
  auto promote_int_to_float = rounding_mode == StrModeTrue;
  auto result_type =
      habana_helpers::DTypeHelper::
          binary_op_with_optional_int_to_float_promotion(
              {self, other}, !promote_int_to_float, c10::nullopt, false)
              .get_common_dtype();
  switch (result_type) {
    case torch::kBFloat16:
    case torch::kFloat32:
    case torch::kFloat64:
      return true;
    case torch::kHalf: {
      auto device_type{synapse_helpers::HPURegistrar::get_device().type()};
      return device_type == synDeviceGaudi2 || device_type == synDeviceGreco;
    }
    default:
      return false;
  }
}

FALLBACK_CHECK(
    DivTensorFallbackCheck,
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  return DivCommonCheck(self, other, rounding_mode);
}

FALLBACK_CHECK(
    DivScalarFallbackCheck,
    const at::Tensor& self,
    const at::Scalar& other,
    c10::optional<c10::string_view> rounding_mode) {
  return DivCommonCheck(self, other, rounding_mode);
}

template <>
LazyDiv<at::Tensor>::LazyDiv(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  c10::optional<std::string> rounding_mode =
      inputs[2].toOptional<std::string>();
  TORCH_CHECK(
      !rounding_mode.has_value() or (*rounding_mode == StrModeTruncate) or
          (*rounding_mode == StrModeFloor),
      "div expected rounding_mode to be one of None, '",
      StrModeTruncate,
      "', or '",
      StrModeFloor,
      "' "
      "but found '",
      *rounding_mode,
      "'");
}

template <>
at::Tensor LazyDiv<at::Tensor>::get_result_overrideable() {
  auto inputs = LazyOp<at::Tensor>::get_inputs();
  auto self = inputs[0].toTensor();
  auto other = inputs[1].toTensor();

  c10::optional<std::string> rounding_mode =
      inputs[2].toOptional<std::string>();

  const std::string strRroundingMode = rounding_mode.value_or(StrModeTrue);
  auto promote_int_to_float = strRroundingMode == StrModeTrue;
  auto dtype_helper = habana_helpers::DTypeHelper::
      binary_op_with_optional_int_to_float_promotion(
          inputs, promote_int_to_float, c10::nullopt, false);

  c10::ScalarType result_dtype = dtype_helper.get_result_dtype();

  auto shape_out = BinaryOperator::compute_output_shape(self, other);

  auto result = habana_lazy::empty_hpu_lazy(
      shape_out,
      self.options().dtype(result_dtype),
      self.suggest_memory_format(),
      false);

  return result;
}

void DivRoundModeOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  at::Tensor other;
  if (stack.at(1).isTensor()) {
    other = stack_tensor(stack, 1);
  }
  std::string rounding_mode =
      stack[2].isNone() ? StrModeTrue : stack[2].toStringRef();

  std::vector<at::Tensor> tensors = {self, other};

  // Check if mode is other than default "true", i.e. "floor" or "trunc"
  bool bOtherThanTrueMode = (StrModeTrue != rounding_mode);

  // Find the result type
  auto dtype_helper = habana_helpers::DTypeHelper::
      binary_op_with_optional_int_to_float_promotion(
          stack, !bOtherThanTrueMode, c10::nullopt, false);

  auto final_result_type = dtype_helper.get_result_dtype();
  if (stack.at(1).isScalar()) {
    SetScalarType(final_result_type);
  }

  auto shape_out = stack.at(1).isScalar()
      ? self.sizes().vec()
      : BinaryOperator::compute_output_shape(self, other);

  std::vector<synTensor> binaryop_inputs{syn_in(0), syn_in(1)};

  // Handle integral cases differently using div_mod, else floating point
  // convertion yields error after truncation in some cases.
  if (bOtherThanTrueMode && (c10::isIntegralType(final_result_type, true))) {
    size_t size = 0;

    // The second argument of "FillDivModParams", pyCompatible is false
    // for 'trunc' mode and true for 'floor' case
    const auto& params =
        FillDivModParams(size, (StrModeFloor == rounding_mode));

    const std::string opStringSuffix =
        habana_helpers::name_suffix_from_type(final_result_type);
    auto output = GetDivModOutput(
        this,
        graph,
        syn_in(0),
        syn_in(1),
        (StrModeFloor == rounding_mode),
        shape_out,
        final_result_type,
        DIV_MODE_OUTPUT_TYPE::QUOTIENT);
    syn_out(0) = std::move(output[0]);

    return;
  } else { // if (isIntegralType(final_result_type, true))

    // Computation is always done in float or bfloat16
    at::ScalarType computation_type = dtype_helper.get_common_dtype();
    const std::string opStringSuffix =
        "_fwd_" + habana_helpers::name_suffix_from_type(computation_type);

    // Initialization
    const unsigned int cNoOfInputTensors = stack.at(1).isScalar() ? 1 : 2;
    std::vector<synapse_helpers::tensor> divOp, makeIntegerOp;
    std::unique_ptr<synapse_helpers::tensor> cast[cNoOfInputTensors];

    // Convert each tensor to float/bfloat16 (if not already in)
    std::string strNode_type;

    for (unsigned char i = 0; i < cNoOfInputTensors; ++i) {
      if (tensors.at(i).scalar_type() == computation_type) {
        continue;
      }

      cast[i] = std::make_unique<synapse_helpers::tensor>(CastHelper(
          graph,
          syn_in(i),
          stack_tensor(stack, i).sizes(),
          tensors.at(i).scalar_type(),
          computation_type));
      binaryop_inputs.at(i) = cast[i]->get();
    }

    // Final cast is required, if result type is not float
    // when those are same, and following flag bNeedToCastFinalResult is false
    // we used computation_type for output type to avoid multiple branches
    bool bNeedToCastFinalResult = (final_result_type != computation_type);

    divOp = BuildOp(
        graph,
        "div_" + habana_helpers::name_suffix_from_type(computation_type),
        binaryop_inputs,
        {{shape_out,
          computation_type,
          bOtherThanTrueMode ? c10::nullopt : c10::make_optional<int>(0)}});
    if (!bOtherThanTrueMode) {
      // when flow reaches here, computation_type is same as final_result_type,
      // so computation_type can be used as div's return type and that is the
      // return type of div_rounding_mode
      syn_out(0) = std::move(divOp[0]);
      return;
    }

    // If in "floor" or "trunc" mode, need to apply that
    makeIntegerOp = BuildOp(
        graph,
        rounding_mode + opStringSuffix,
        {divOp.at(0).get()},
        {{shape_out,
          computation_type,
          bNeedToCastFinalResult ? c10::nullopt : c10::make_optional<int>(0)}});
    if (!bNeedToCastFinalResult) {
      // when flow reaches here, computation_type is same as final_result_type,
      // so computation_type can be used as return type of floor/trunc and that
      // is the return type of div_rounding_mode
      syn_out(0) = std::move(makeIntegerOp[0]);
      return;
    }

    auto castToReturnTypeOp = CastHelper(
        graph,
        makeIntegerOp.at(0).get(),
        shape_out,
        computation_type,
        final_result_type,
        0);
    syn_out(0) = std::move(castToReturnTypeOp);
  } // else { //if (isIntegralType(final_result_type, true))
}

} // namespace habana
