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
#include "generated/hpu_op.h"
#include "habana_kernels/binary_kernels.h"

// TODO: Need to fetch pytorch default type dynamically
// pytorch default type and handle other types as default type
// Default pytorch dtype is assumed to be c10::ScalarType::Float
// here.
#define PYTORCH_DEFAULT_TYPE c10::ScalarType::Float

// Except bfloat16, all other types are computed in following type
#define COMMON_COMPUTATION_TYPE_TPC c10::ScalarType::Float

// For use in div_rounding_mode
#define StrModeTrue ""
#define StrModeFloor "floor"
#define StrModeTruncate "trunc"

namespace habana {

template <>
LazyDiv<at::Tensor>::LazyDiv(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
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
  // If types of both Tensors are same, then first Tensor's type is used.
  // This was required since this this case, the call  to
  // type_promotion_for_two_tensor_inputs(...) does not change the
  // result_dtype's existing value.
  c10::ScalarType result_dtype = inputs[0].toTensor().scalar_type();

  // TODO: Take this type promo code to a separate function to avoid
  // duplicattion
  int pos = -1;
  habana_helpers::type_promotion_for_two_tensor_inputs(
      inputs, pos, result_dtype);

  // In true mode, div of integral types results in default type
  const std::string strRroundingMode = rounding_mode.value_or(StrModeTrue);
  if ((strRroundingMode == StrModeTrue) and
      (c10::isIntegralType(result_dtype, true))) {
    // TODO: default type is assumed to be Float,
    // need to check using APIs
    result_dtype = PYTORCH_DEFAULT_TYPE;
  }

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

  // Find the result type
  const at::ScalarType& final_result_type = stack.at(1).isScalar()
      ? ComputePromotedScalarType(stack, true)
      : at::result_type(self, other);

  // Computation is always done in float
  const at::ScalarType& computation_type =
      (c10::ScalarType::BFloat16 == final_result_type)
      ? c10::ScalarType::BFloat16
      : COMMON_COMPUTATION_TYPE_TPC;
  const std::string opStringSuffix =
      "_fwd_" + habana_helpers::name_suffix_from_type(computation_type);

  // Initialization
  const unsigned int cNoOfInputTensors = stack.at(1).isScalar() ? 1 : 2;
  std::vector<synapse_helpers::tensor> divOp, cast[cNoOfInputTensors],
      makeIntegerOp, castToReturnTypeOp;

  std::vector<synTensor> binaryop_inputs{syn_in(0), syn_in(1)};

  // Convert each tensor to float/bfloat16 (if not already in)
  std::pair<c10::ScalarType, c10::ScalarType> type_key;
  std::string strNode_type;

  for (unsigned char i = 0; i < cNoOfInputTensors; ++i) {
    if (tensors.at(i).scalar_type() == computation_type) {
      continue;
    }

    type_key = std::make_pair(tensors.at(i).scalar_type(), computation_type);
    auto iter = habana_helpers::cast_map.find(type_key);
    strNode_type = iter->second;
    cast[i] = BuildOp(
        graph,
        strNode_type,
        {syn_in(i)},
        // Sizes of "self" and "other" tensors may differ, so
        // use own size as output size for cast op
        {{stack_tensor(stack, i).sizes(), computation_type}});
    binaryop_inputs.at(i) = cast[i].at(0).get();
  }

  auto shape_out = stack.at(1).isScalar()
      ? self.sizes().vec()
      : BinaryOperator::compute_output_shape(self, other);

  // Check if mode is other than default "true", i.e. "floor" or "trunc"
  bool bOtherThanTrueMode = (StrModeTrue != rounding_mode);

  // Final cast is required, if result type is not float
  bool bNeedToCastFinalResult = (final_result_type != computation_type);

  divOp = BuildOp(
      graph,
      "div" + opStringSuffix,
      binaryop_inputs,
      {{shape_out,
        computation_type,
        bOtherThanTrueMode ? c10::nullopt : c10::make_optional<int>(0)}});
  if (!bOtherThanTrueMode) {
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
    syn_out(0) = std::move(makeIntegerOp[0]);
    return;
  }

  type_key = std::make_pair(computation_type, final_result_type);
  auto iter = habana_helpers::cast_map.find(type_key);
  strNode_type = iter->second;
  castToReturnTypeOp = BuildOp(
      graph,
      strNode_type,
      {makeIntegerOp.at(0).get()},
      {{shape_out, final_result_type, 0}});
  syn_out(0) = std::move(castToReturnTypeOp[0]);
}

} // namespace habana
