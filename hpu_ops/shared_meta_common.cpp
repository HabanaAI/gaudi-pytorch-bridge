/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/shared_meta_common.h"
#include <unordered_set>
namespace habana {

// if all integers are not supported enter only torch::kInt32
static std::unordered_map<std::string, std::set<at::ScalarType>>
    foreachOpsUnsupportedDtypes = {
        {"acos_fwd", {torch::kInt32, torch::kFloat16, torch::kBFloat16}},
        {"asin_fwd", {torch::kInt32, torch::kFloat16, torch::kBFloat16}},
        {"atan_fwd", {torch::kInt32, torch::kFloat16, torch::kBFloat16}},
        {"ceil_fwd", {torch::kInt32}},
        {"cos_fwd", {torch::kInt32}},
        {"cosh_fwd", {torch::kInt32, torch::kFloat16, torch::kBFloat16}},
        {"erf_fwd", {torch::kInt32, torch::kFloat16}},
        {"exp_fwd", {torch::kInt32}},
        {"expm1_fwd", {torch::kInt32, torch::kFloat16, torch::kBFloat16}},
        {"floor_fwd", {torch::kInt32}},
        {"gammaln_fwd", {torch::kInt32, torch::kFloat16, torch::kBFloat16}},
        {"log_fwd", {torch::kInt32}},
        {"log10_fwd", {torch::kInt32}},
        {"log1p_fwd", {torch::kInt32}},
        {"log2_fwd", {torch::kInt32}},
        {"neg_fwd", {torch::kInt16, torch::kInt8}},
        {"reciprocal_fwd", {torch::kInt32}},
        {"round_fwd", {torch::kInt32}},
        {"sigmoid_fwd", {torch::kInt32}},
        {"sin_fwd", {torch::kInt32}},
        {"sinh_fwd", {torch::kInt32, torch::kFloat16, torch::kBFloat16}},
        {"sqrt_fwd", {torch::kInt32}},
        {"tan_fwd", {torch::kInt32, torch::kFloat16, torch::kBFloat16}},
        {"tanh_fwd", {torch::kInt32}},
        {"trunc_fwd", {torch::kInt32}},
};

SharedMetaDataVector Input0SharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);

  SharedMetaData meta{guid};
  meta.inputs_data = {{input.dim(), input.scalar_type()}};
  meta.outputs_data = {meta.inputs_data[0]};

  return {meta};
}

SharedMetaDataVector Input0ToOut0And1SharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);

  SharedMetaData meta{guid};
  SharedMetaTensor inOutTensor = {input.dim(), input.scalar_type()};
  meta.inputs_data = {inOutTensor};
  meta.outputs_data = {inOutTensor, inOutTensor};

  return {meta};
}

SharedMetaDataVector AdaptiveBwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& input = stack_tensor(stack, 1);

  SharedMetaData meta{guid};
  meta.inputs_data = {
      {grad.dim(), grad.scalar_type()}, {input.dim(), input.scalar_type()}};
  meta.outputs_data = {{input.dim(), grad.scalar_type()}};

  return {meta};
}

SharedMetaDataVector AvgPoolBwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& input = stack_tensor(stack, 1);

  SharedMetaData meta{guid};
  meta.inputs_data = {{grad.dim(), grad.scalar_type()}};
  meta.outputs_data = {{input.dim(), input.scalar_type()}};

  return {meta};
}

SharedMetaDataVector FillCumSumProdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);
  at::ScalarType dtype =
      stack.at(2).isNone() ? input.scalar_type() : stack.at(2).toScalarType();

  if (habana_helpers::is_downcast_to_int_needed(dtype))
    dtype = at::ScalarType::Int;
  else if (dtype == at::ScalarType::Double)
    dtype = at::ScalarType::Float;
  else if (
      dtype == at::ScalarType::Bool || dtype == at::ScalarType::Char ||
      dtype == at::ScalarType::Byte)
    dtype = at::ScalarType::Int;

  SharedMetaData meta{guid};
  meta.inputs_data = {{input.dim(), dtype}};
  meta.outputs_data = {{input.dim(), dtype}};

  return {meta};
}

SharedMetaDataVector IsFiniteInfNanSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();

  if (c10::isIntegralType(dtype, true))
    dtype = c10::ScalarType::Int;

  SharedMetaData meta{guid};
  meta.inputs_data = {{rank, dtype}};
  meta.outputs_data = {{rank, torch::kBool}};

  return {meta};
}

SharedMetaDataVector RoundingSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto input = stack.at(0).toTensor();
  auto rank = input.dim();
  auto dtype = input.scalar_type();

  SharedMetaData roundingMeta;
  roundingMeta.guid = c10::isIntegralType(dtype, true) ? "identity" : guid;

  SharedMetaTensor inOutTensor = {rank, dtype};
  roundingMeta.inputs_data = {inOutTensor};
  roundingMeta.outputs_data = {inOutTensor};
  return {roundingMeta};
}

SharedMetaDataVector UnaryForeachSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto tensors = stack.at(0).toTensorList();
  auto tensorsSize = tensors.size();
  SharedMetaDataVector metaVec;
  metaVec.resize(tensorsSize);
  for (size_t i = 0; i < tensorsSize; i++) {
    const at::Tensor& tensor = tensors[i];
    auto rank = tensor.dim();
    auto inputType = tensor.scalar_type();

    auto guidIt = foreachOpsUnsupportedDtypes.find(guid);
    if (guidIt != std::end(foreachOpsUnsupportedDtypes)) {
      bool isInputTypeUnsupported =
          guidIt->second.find(inputType) != std::end(guidIt->second);
      if (isIntegralType(inputType, true)) {
        bool isI32Unsupported =
            guidIt->second.find(torch::kInt32) != std::end(guidIt->second);
        if (isI32Unsupported)
          inputType = torch::kFloat32;
        else
          inputType = isInputTypeUnsupported ? torch::kInt32 : inputType;
      } else if (isInputTypeUnsupported) {
        inputType = torch::kFloat32;
      }
    }

    auto outputType = inputType;
    SharedMetaData foreachMeta{guid};
    foreachMeta.inputs_data = {{rank, inputType}};
    foreachMeta.outputs_data = {{rank, outputType}};
    metaVec[i] = foreachMeta;
  }
  return metaVec;
}

SharedMetaDataVector CompareSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto self = stack_tensor(stack, 0);
  auto other = stack.at(1);
  auto selfRank = self.dim();
  auto otherRank = other.isScalar() ? 1 : other.toTensor().dim();
  auto outputRank = std::max(selfRank, otherRank);
  auto inputType = habana_helpers::DTypeHelper::get_compute_dtype(
      {self, other},
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  if ((guid == "less" || guid == "less_fwd") &&
      inputType == c10::ScalarType::Short)
    inputType = c10::ScalarType::Int;

  SharedMetaData compareSharedMeta{guid};
  compareSharedMeta.inputs_data = {
      {selfRank, inputType}, {otherRank, inputType}};
  compareSharedMeta.outputs_data = {{outputRank, c10::ScalarType::Bool}};
  return {compareSharedMeta};
}

SharedMetaDataVector ForeachCompoundSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& selfs = stack.at(0).toTensorList();
  const auto& tensors1 = stack.at(1).toTensorList();
  const auto& tensors2 = stack.at(2).toTensorList();
  const auto& value = stack.at(3);
  const bool isValueTensor = value.isTensor();
  const auto selfsSize = selfs.size();
  int maxNoOfNodesPerIteration = isValueTensor ? 3 : 1;
  SharedMetaDataVector metaVec;
  metaVec.reserve(selfsSize * maxNoOfNodesPerIteration);
  for (size_t i = 0; i < selfsSize; ++i) {
    const auto& self = selfs[i];
    const auto& tensor1 = tensors1[i];
    const auto& tensor2 = tensors2[i];
    const auto selfRank = self.dim();
    const auto tensor1Rank = tensor1.dim();
    const auto tensor2Rank = tensor2.dim();
    const auto selfDtype = self.scalar_type();
    at::ScalarType dtype =
        at::promote_types(selfDtype, at::result_type(tensor1, tensor2));
    const int outputRank =
        std::max(selfRank, std::max(tensor1Rank, tensor2Rank));
    bool isAddcdiv = guid == "addcdiv_fwd";
    const bool isOutputIntegral = c10::isIntegralType(dtype, true);
    dtype = (isAddcdiv && isOutputIntegral) ? torch::kFloat32 : dtype;
    c10::optional<SharedMetaData> floorSharedMeta = c10::nullopt;

    if (isValueTensor) {
      auto valueTensor = value.toTensor();
      auto valueRank = valueTensor.dim();
      auto valueDtype = valueTensor.scalar_type();
      SharedMetaData sliceAxisSharedMeta{"slice_axis"};
      sliceAxisSharedMeta.inputs_data.emplace_back(valueRank, valueDtype);
      sliceAxisSharedMeta.outputs_data.emplace_back(1, valueDtype);
      metaVec.push_back(sliceAxisSharedMeta);

      if (c10::isFloatingType(valueDtype) && isOutputIntegral) {
        floorSharedMeta = {"floor_fwd"};
        sliceAxisSharedMeta.inputs_data = sliceAxisSharedMeta.outputs_data;
        sliceAxisSharedMeta.outputs_data = sliceAxisSharedMeta.inputs_data;
        metaVec.push_back(sliceAxisSharedMeta);
      }
    }

    SharedMetaData compositeSharedMeta{guid};
    compositeSharedMeta.inputs_data = {
        {selfRank, dtype}, {tensor1Rank, dtype}, {tensor2Rank, dtype}};
    if (floorSharedMeta.has_value())
      compositeSharedMeta.inputs_data.emplace_back(1, dtype);

    compositeSharedMeta.outputs_data.emplace_back(outputRank, dtype);
    metaVec.push_back(compositeSharedMeta);
  }
  return metaVec;
}

SharedMetaDataVector BoolCastSharedMeta(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();
  SharedMetaDataVector metaVec = {};
  SharedMetaData equalFwdMeta{"equal_fwd"};
  equalFwdMeta.inputs_data = {{rank, dtype}, {rank, dtype}};
  equalFwdMeta.outputs_data = {{rank, at::kBool}};
  metaVec.push_back(equalFwdMeta);

  SharedMetaData notFwdMeta("not_fwd");
  notFwdMeta.inputs_data = equalFwdMeta.outputs_data;
  notFwdMeta.outputs_data = equalFwdMeta.outputs_data;
  metaVec.push_back(notFwdMeta);
  return metaVec;
}

SharedMetaDataVector LogicalBinarySharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto self = stack.at(0).toTensor();
  auto other = stack.at(1).toTensor();
  auto selfRank = self.dim();
  auto selfDtype = self.scalar_type();
  auto otherDtype = other.scalar_type();
  const bool isI16 = selfDtype == c10::ScalarType::Short ||
      otherDtype == c10::ScalarType::Short;
  const bool isI64 =
      selfDtype == c10::ScalarType::Long || otherDtype == c10::ScalarType::Long;
  const bool isI16orI64 = isI16 || isI64;
  bool promoteToCommonType = false;

  if ((selfDtype == c10::ScalarType::Char &&
       otherDtype == c10::ScalarType::Byte) ||
      (selfDtype == c10::ScalarType::Byte &&
       otherDtype == c10::ScalarType::Char)) {
    selfDtype = c10::ScalarType::Byte;
    otherDtype = c10::ScalarType::Byte;
  } else if (selfDtype != otherDtype) {
    auto isSelfIntegral = c10::isIntegralType(selfDtype, false);
    auto isOtherIntegral = c10::isIntegralType(otherDtype, false);
    if (guid == "and" && !isI16) {
      promoteToCommonType = true;
    } else if ((guid == "or" || "xor") && !isI16orI64) {
      if (!(isSelfIntegral || isOtherIntegral) ||
          ((isSelfIntegral ^ isOtherIntegral) &&
           ((c10::elementSize(selfDtype) == 1 ||
             c10::elementSize(otherDtype) == 1)))) {
        promoteToCommonType = true;
      }
    }
  }

  if (promoteToCommonType) {
    auto computeDtype = habana_helpers::DTypeHelper::get_compute_dtype(
        {self, other},
        c10::nullopt,
        habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
        false,
        c10::nullopt,
        false,
        false);
    selfDtype = computeDtype;
    otherDtype = computeDtype;
  }

  SharedMetaData logicalBinaryMeta{guid};
  logicalBinaryMeta.inputs_data = {
      {selfRank, selfDtype}, {other.dim(), otherDtype}};
  logicalBinaryMeta.outputs_data = {{selfRank, at::kBool}};
  return {logicalBinaryMeta};
}

} // namespace habana
