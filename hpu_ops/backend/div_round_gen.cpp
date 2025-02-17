/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "hpu_ops/common/div_round_gen.h"
#include "generated/backend/div.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/binary_kernels.h"
#include "hpu_ops/div_mod_util.h"

namespace habana {

OutputMetaDataVector DivModeMeta(const at::Stack& stack) {
  OutputMetaData meta{};
  const auto& self = stack_tensor(stack, 0);
  if (stack[1].isScalar()) {
    meta.shape = self.sizes().vec();
  } else {
    meta.shape = at::infer_size(self.sizes(), stack_tensor(stack, 1).sizes());
  }
  meta.dtype = GetResultDtype(stack, stack[2].isNone());

  return {meta};
}

SharedMetaDataVector DivModeSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack.at(0);
  auto selfTensor = self.toTensor();
  auto selfRank = selfTensor.dim();
  auto other = stack.at(1);
  int64_t otherRank = other.isTensor() ? other.toTensor().dim() : 1;
  auto outputRank = std::max(selfRank, otherRank);

  auto isRoundingModeNone = stack.at(2).isNone();
  auto resultType = GetResultDtype(stack, isRoundingModeNone);
  if (!isRoundingModeNone && c10::isIntegralType(resultType, true)) {
    SharedMetaData divModMeta{"div_mod_fwd"};
    divModMeta.inputs_data = {{selfRank, resultType}, {otherRank, resultType}};
    divModMeta.outputs_data = {
        {outputRank, resultType}, {outputRank, resultType}};
    return {divModMeta};
  } else {
    auto commonType = GetCommonDtype(stack, isRoundingModeNone);
    std::string guid = "div";
    if (commonType == at::ScalarType::Float &&
        IS_ENV_FLAG_DEFINED_NEW(PT_HPU_ENABLE_DIV_PRECISE) &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DIV_PRECISE))
      guid = "div_precise";

    SharedMetaData divMeta{guid};
    divMeta.inputs_data = {{selfRank, commonType}, {otherRank, commonType}};
    divMeta.outputs_data = {{outputRank, commonType}};
    return {divMeta};
  }
}

std::vector<synapse_helpers::tensor> DivCommonFunction(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<synTensor> binaryop_inputs,
    const c10::optional<c10::string_view>& rounding_mode) {
  // Check if mode is other than None, i.e. "floor" or "trunc"
  bool isNotNone = rounding_mode.has_value();

  // Find the result type
  auto final_result_type = GetResultDtype(stack, !isNotNone);

  const at::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> shape_out = self.sizes().vec();
  // If second input is tensor compute out shape based on both inputs
  if (stack.at(1).isTensor()) {
    const at::Tensor& other = stack_tensor(stack, 1);
    shape_out = BinaryOperator::compute_output_shape(self, other);
  }

  // Handle integral cases differently using div_mod, else floating point
  // convertion yields error after truncation in some cases.
  if (isNotNone && (c10::isIntegralType(final_result_type, true))) {
    // The second argument of "FillDivModParams", pyCompatible is false
    // for 'trunc' mode and true for 'floor' case
    return GetDivModOutput(
        op,
        graph,
        binaryop_inputs[0],
        binaryop_inputs[1],
        (StrModeFloor == rounding_mode),
        std::move(shape_out),
        DIV_MODE_OUTPUT_TYPE::QUOTIENT);
  } else { // if (isIntegralType(final_result_type, true))

    // Computation is always done in float or bfloat16
    at::ScalarType computation_type = GetCommonDtype(stack, !isNotNone);

    // Initialization
    const unsigned cNoOfInputs = 2;
    std::vector<synapse_helpers::tensor> divOp, makeIntegerOp;
    std::unique_ptr<synapse_helpers::tensor> cast[cNoOfInputs];

    // Convert each tensor to float/bfloat16 (if not already in)
    std::string strNode_type;

    for (unsigned char i = 0; i < cNoOfInputs; ++i) {
      auto stack_input = stack.at(i);
      if (stack_input.isTensor() &&
          stack_input.toTensor().scalar_type() != computation_type) {
        auto tensor = stack_tensor(stack, i);
        cast[i] =
            std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
                op,
                graph,
                binaryop_inputs[i],
                tensor.sizes(),
                tensor.scalar_type(),
                computation_type));
        binaryop_inputs.at(i) = cast[i]->get();
      }
    }

    // Final cast is required, if result type is not float
    // when those are same, and following flag bNeedToCastFinalResult is false
    // we used computation_type for output type to avoid multiple branches
    bool bNeedToCastFinalResult = (final_result_type != computation_type);

    auto guid = op->GetGuid();
    if (computation_type == at::ScalarType::Float) {
      // Update the div guid with precise based on env and rounding mode
      guid = update_div_guid_with_precise(guid, isNotNone);
    }
    divOp = OpBackend::BuildNode(
        op,
        graph,
        {update_guid_dtype(guid, computation_type),
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
        {get_guid_with_precision(std::string(*rounding_mode), computation_type),
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
