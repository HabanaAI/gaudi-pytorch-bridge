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

#include "hpu_ops/common/div_round_gen.h"
#include "generated/backend/div.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/binary_kernels.h"
#include "hpu_ops/div_mod_util.h"

namespace habana {

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
