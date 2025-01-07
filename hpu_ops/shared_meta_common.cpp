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

#include "hpu_ops/shared_meta_common.h"
#include <unordered_set>
#include "backend/helpers/runtime_config.h"
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
  const auto tensors = stack.at(0).toTensorList();
  const auto tensorsSize = tensors.size();
  SharedMetaDataVector metaVec;
  metaVec.reserve(tensorsSize);
  for (size_t i = 0; i < tensorsSize; i++) {
    const at::Tensor& tensor = tensors[i];
    const auto rank = tensor.dim();
    if (guid != "constant" || (guid == "constant" && rank > 1)) {
      auto dtype = tensor.scalar_type();
      const auto guidIt = foreachOpsUnsupportedDtypes.find(guid);
      if (guidIt != std::end(foreachOpsUnsupportedDtypes)) {
        bool isDtypeUnsupported =
            guidIt->second.find(dtype) != std::end(guidIt->second);
        if (isIntegralType(dtype, true)) {
          bool isI32Unsupported =
              guidIt->second.find(torch::kInt32) != std::end(guidIt->second);
          if (isI32Unsupported)
            dtype = torch::kFloat32;
          else
            dtype = isDtypeUnsupported ? torch::kInt32 : dtype;
        } else if (isDtypeUnsupported) {
          dtype = torch::kFloat32;
        }
      }

      SharedMetaData foreachMeta{guid};
      foreachMeta.inputs_data = {{rank, dtype}};
      foreachMeta.outputs_data = {{rank, dtype}};
      metaVec.push_back(foreachMeta);
    }
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

SharedMetaDataVector BoolCastSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
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

SharedMetaDataVector AminAmaxSharedMeta(
    const at::Stack& stack,
    const std::string& guid,
    habana_helpers::HabanaExecutionMode executionMode) {
  const auto self = stack.at(0).toTensor();
  const bool keepDim = stack.size() >= 3 ? stack.at(2).toBool() : false;
  const auto rank = self.dim();
  auto inputDtype = self.scalar_type();

  SharedMetaDataVector metaVec;
  if (inputDtype == c10::ScalarType::Bool) {
    metaVec = BoolCastSharedMeta({self}, executionMode);
  }

  if (c10::isIntegralType(inputDtype, true)) {
    inputDtype = c10::ScalarType::Int;
  }
  auto outputDtype = inputDtype;

  auto outputRank = rank;
  if (!keepDim) {
    int dimNum = 0;
    if (stack.size() >= 2) {
      const auto dim = stack.at(1);
      if (dim.isIntList())
        dimNum = dim.isNone() ? 0 : dim.toIntVector().size();
      else if (dim.isInt())
        dimNum = 1;
    }
    outputRank = dimNum == 0 || outputRank == 0 ? 0 : outputRank - dimNum;
  }

  SharedMetaData reduceMaxMultiDimFwdMeta(guid);
  reduceMaxMultiDimFwdMeta.options.allowLongType = true;
  reduceMaxMultiDimFwdMeta.inputs_data = {{rank, inputDtype}};
  reduceMaxMultiDimFwdMeta.outputs_data.emplace_back(outputRank, outputDtype);
  reduceMaxMultiDimFwdMeta.outputs_data.emplace_back(
      outputRank, c10::ScalarType::Long);
  metaVec.push_back(reduceMaxMultiDimFwdMeta);
  return metaVec;
}

SharedMetaDataVector BinaryWithAlphaSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto self = stack.at(0);
  auto other = stack.at(1);
  auto selfTensor = self.toTensor();
  auto selfRank = selfTensor.dim();
  auto otherRank = other.isTensor() ? other.toTensor().dim() : 1;
  auto outputRank = std::max(selfRank, otherRank);

  at::ScalarType outputType;
  if (other.isTensor()) {
    const at::Tensor& otherTensor = other.toTensor();
    outputType = at::result_type(selfTensor, otherTensor);
  } else {
    const auto& otherScalar = other.toScalar();
    outputType = at::result_type(selfTensor, otherScalar);
  }

  const auto& alpha = stack.at(2).toScalar();
  SharedMetaDataVector metaVec;
  if (alpha.equal(1)) {
    // This node will only appear in eager mode but there is no way to
    // distinguish mode here so both possibilities should be added to
    // verification
    std::string opName;
    SharedMetaData binaryKernelMeta;
    binaryKernelMeta.inputs_data = {
        {selfRank, outputType}, {otherRank, outputType}};
    binaryKernelMeta.outputs_data = {{outputRank, outputType}};
    if (guid == "add") {
      binaryKernelMeta.guid = "add";
    } else {
      if (guid == "rsub") {
        binaryKernelMeta.inputs_data = {
            {otherRank, outputType}, {selfRank, outputType}};
      }
      binaryKernelMeta.guid = "sub";
    }
    metaVec.push_back(binaryKernelMeta);
  }
  SharedMetaData binaryWithAlphaMeta{"binary_with_alpha_fwd"};
  binaryWithAlphaMeta.inputs_data = {
      {selfRank, outputType}, {otherRank, outputType}};
  binaryWithAlphaMeta.outputs_data = {{outputRank, outputType}};
  metaVec.push_back(binaryWithAlphaMeta);
  return metaVec;
}

SharedMetaDataVector BitwiseLogicalSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto self = stack.at(0);
  auto other = stack.at(1);
  auto dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      {self, self},
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  auto inputRank = self.isTensor() ? self.toTensor().dim() : 1;
  auto otherRank = other.isTensor() ? other.toTensor().dim() : 1;
  auto outputRank = std::max(inputRank, otherRank);

  SharedMetaData bitwiseSharedMeta{guid};
  bitwiseSharedMeta.inputs_data = {{inputRank, dtype}, {otherRank, dtype}};
  bitwiseSharedMeta.outputs_data.emplace_back(outputRank, dtype);
  return {bitwiseSharedMeta};
}

SharedMetaDataVector TopkSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack.at(0).toTensor();

  SharedMetaData topkMeta("topk");
  topkMeta.inputs_data.emplace_back(self.dim(), self.scalar_type());
  topkMeta.outputs_data.emplace_back(self.dim(), self.scalar_type());
  topkMeta.outputs_data.emplace_back(self.dim(), c10::ScalarType::Int);

  return {topkMeta};
}

SharedMetaDataVector RandomSeedTensorInputSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto self = stack_tensor(stack, 0);
  auto seed = stack.at(3);
  SharedMetaTensor seedSharedTensor = {1, c10::ScalarType::Int};
  if (seed.isTensor()) {
    const auto seedTensor = seed.toTensor();
    seedSharedTensor = {seedTensor.dim(), seedTensor.scalar_type()};
  }

  const auto isUniform =
      guid.find("philox_random_uniform") != std::string::npos;
  auto computeDtype = self.scalar_type();
  SharedMetaData randomSharedMeta{guid};
  if (!isUniform) {
    randomSharedMeta.inputs_data.push_back(
        createOptionalNotPresentSharedMetaTensor());
    if (computeDtype != c10::ScalarType::BFloat16)
      computeDtype = c10::ScalarType::Float;
  }

  randomSharedMeta.inputs_data.push_back(seedSharedTensor);
  if (isUniform)
    randomSharedMeta.inputs_data.push_back(randomSharedMeta.inputs_data[0]);

  randomSharedMeta.outputs_data.emplace_back(self.dim(), computeDtype);
  return {randomSharedMeta};
}

SharedMetaDataVector MaxPoolWithIndicesFwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& self = stack_tensor(stack, 0);
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();
  auto indexType = c10::ScalarType::Long;

  SharedMetaData maxPoolWithIndicesSharedMeta{guid};
  maxPoolWithIndicesSharedMeta.inputs_data.emplace_back(rank, dtype);
  if (guid.find("maxpool_3d") != std::string::npos) {
    switch (dtype) {
      case c10::ScalarType::BFloat16:
      case c10::ScalarType::Half:
        indexType = c10::ScalarType::Short;
        break;
      default:
        indexType = c10::ScalarType::Byte;
        break;
    }
    maxPoolWithIndicesSharedMeta.outputs_data = {
        {rank, indexType}, {rank, dtype}};
  } else {
    maxPoolWithIndicesSharedMeta.options.allowLongType = true;
    maxPoolWithIndicesSharedMeta.outputs_data = {
        {rank, dtype}, {rank, indexType}};
  }

  return {maxPoolWithIndicesSharedMeta};
}

SharedMetaDataVector MaxPoolWithIndicesBwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& self = stack_tensor(stack, 1);
  const auto& indices = stack_tensor(stack, 7);
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();
  auto indexType = c10::ScalarType::Long;
  SharedMetaData maxPoolWithIndicesSharedMeta{guid};
  maxPoolWithIndicesSharedMeta.inputs_data.emplace_back(
      grad.dim(), grad.scalar_type());
  bool isMaxPool3d = guid.find("maxpool_3d") != std::string::npos;
  if (isMaxPool3d) {
    switch (dtype) {
      case c10::ScalarType::BFloat16:
      case c10::ScalarType::Half:
        indexType = c10::ScalarType::Short;
        break;
      default:
        indexType = c10::ScalarType::Byte;
        break;
    }

    // optional not present tensors required for non TF version
    auto optionalNotPresentTensor = createOptionalNotPresentSharedMetaTensor();
    maxPoolWithIndicesSharedMeta.inputs_data.push_back(
        optionalNotPresentTensor);
    maxPoolWithIndicesSharedMeta.inputs_data.push_back(
        optionalNotPresentTensor);
  } else {
    maxPoolWithIndicesSharedMeta.inputs_data.emplace_back(rank, dtype);
    maxPoolWithIndicesSharedMeta.options.allowLongType = true;
  }
  maxPoolWithIndicesSharedMeta.inputs_data.emplace_back(
      indices.dim(), indexType);
  maxPoolWithIndicesSharedMeta.outputs_data = {{rank, dtype}};

  return {maxPoolWithIndicesSharedMeta};
}

SharedMetaDataVector EmptySharedMeta(
    const at::Stack&,
    habana_helpers::HabanaExecutionMode) {
  // op doesn't call any kernels or [SW-205149] return empty vector because
  // shape tensor validation will block shape agnostic flow
  return {};
}

SharedMetaDataVector MatmulSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto& other = stack_tensor(stack, 1);
  const auto dtype = self.scalar_type();
  auto selfRank = self.dim();
  auto otherRank = other.dim();
  const bool isBiasPresentForBmm =
      (((stack.size() == 3) || (stack.size() == 5)) &&
       (stack.at(2).toTensor().dim() == 1));
  bool addBias = false;
  int64_t outputRank = std::max(selfRank, otherRank);
  std::string guid;
  const auto matmul3d2dReshapeEnabled =
      habana_helpers::IsMatmul3d2dReshapeEnabled();
  if ((selfRank == 1 && otherRank == 1) || (selfRank == 2 && otherRank == 1) ||
      (selfRank == 1 && otherRank == 2) || (selfRank == 2 && otherRank == 2) ||
      (matmul3d2dReshapeEnabled && selfRank == 3 && otherRank == 2)) {
    selfRank = 2;
    otherRank = 2;
    outputRank = 2;
    guid = "gemm";
    if (matmul3d2dReshapeEnabled && selfRank == 3 && otherRank == 2)
      addBias = isBiasPresentForBmm;
  } else {
    guid = "batch_gemm";
    if (selfRank >= 3 && otherRank == 1) {
      otherRank = 2;
    } else if ((selfRank == 1 || selfRank == 2) && otherRank >= 3) {
      selfRank = 2;
      addBias = isBiasPresentForBmm;
    } else if (
        (selfRank == 4 && otherRank == 3) ||
        (selfRank == 3 && otherRank == 4)) {
      selfRank = 4;
      otherRank = 4;
    } else if (
        (selfRank >= 1 && otherRank >= 1) &&
        (selfRank >= 3 || otherRank >= 3)) {
      addBias = isBiasPresentForBmm;
    }
  }

  SharedMetaData gemmSharedMeta{guid};
  gemmSharedMeta.inputs_data = {{selfRank, dtype}, {otherRank, dtype}};
  if (addBias) {
    const auto& bias = stack_tensor(stack, 2);
    gemmSharedMeta.inputs_data.emplace_back(bias.dim(), bias.scalar_type());
  }
  gemmSharedMeta.outputs_data.emplace_back(outputRank, dtype);

  return {gemmSharedMeta};
}

SharedMetaDataVector StridedViewSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto dtype = self.scalar_type();
  const auto& sizes = stack.at(1);

  SharedMetaData stridedViewSharedMeta{"strided_view"};
  stridedViewSharedMeta.inputs_data.emplace_back(self.dim(), dtype);
  int64_t outputRank;
  if (sizes.isTensor()) {
    const auto& sizesTensor = stack_tensor(stack, 1);
    outputRank = sizesTensor.dim();
    stridedViewSharedMeta.inputs_data.emplace_back(
        outputRank, sizesTensor.scalar_type());
  } else {
    outputRank = sizes.toListRef().size();
  }
  stridedViewSharedMeta.outputs_data.emplace_back(outputRank, dtype);

  return {stridedViewSharedMeta};
}

} // namespace habana
