/**
 * Copyright (c) 2021-2026 Intel Corporation
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
#include "backend/habana_operator.h"
#include "backend/helpers/runtime_config.h"
#include "hpu_ops/backend/foreach.h"

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

  SharedMetaDataVector metaVec;
  metaVec.reserve(1);
  auto& meta = metaVec.emplace_back(guid);
  meta.inputs_data = {{input.dim(), input.scalar_type()}};
  meta.outputs_data = {meta.inputs_data[0]};

  return metaVec;
}

SharedMetaDataVector Input0ToOut0And1SharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);

  SharedMetaDataVector metaVec;
  metaVec.reserve(1);
  auto& meta = metaVec.emplace_back(guid);
  SharedMetaTensor inOutTensor = {input.dim(), input.scalar_type()};
  meta.inputs_data = {inOutTensor};
  meta.outputs_data = {inOutTensor, inOutTensor};

  return metaVec;
}

SharedMetaDataVector AdaptiveBwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& input = stack_tensor(stack, 1);

  SharedMetaDataVector metaVec;
  metaVec.reserve(1);
  auto& meta = metaVec.emplace_back(guid);
  meta.inputs_data = {
      {grad.dim(), grad.scalar_type()}, {input.dim(), input.scalar_type()}};
  meta.outputs_data = {{input.dim(), grad.scalar_type()}};

  return metaVec;
}

SharedMetaDataVector AvgPoolBwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& input = stack_tensor(stack, 1);

  SharedMetaDataVector metaVec;
  metaVec.reserve(1);
  auto& meta = metaVec.emplace_back(guid);
  meta.inputs_data = {{grad.dim(), grad.scalar_type()}};
  meta.outputs_data = {{input.dim(), input.scalar_type()}};

  return metaVec;
}

SharedMetaDataVector FillCumSumProdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);
  at::ScalarType dtype =
      stack.at(2).isNone() ? input.scalar_type() : stack.at(2).toScalarType();

  if (habana_helpers::is_downcast_to_int_needed(dtype) ||
      dtype == at::ScalarType::Bool || dtype == at::ScalarType::Char ||
      dtype == at::ScalarType::Byte) {
    dtype = at::ScalarType::Int;
  } else if (dtype == at::ScalarType::Double) {
    dtype = at::ScalarType::Float;
  }

  SharedMetaDataVector metaVec;
  metaVec.reserve(1);
  auto& meta = metaVec.emplace_back(guid);
  meta.inputs_data = {{input.dim(), dtype}};
  meta.outputs_data = {{input.dim(), dtype}};

  return metaVec;
}

SharedMetaDataVector IsFiniteInfNanSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();

  if (c10::isIntegralType(dtype, true)) {
    dtype = c10::ScalarType::Int;
  }
  SharedMetaDataVector metaVec;
  metaVec.reserve(1);
  auto& meta = metaVec.emplace_back(guid);
  meta.inputs_data = {{rank, dtype}};
  meta.outputs_data = {{rank, torch::kBool}};

  return metaVec;
}

SharedMetaDataVector RoundingSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto input = stack.at(0).toTensor();
  auto rank = input.dim();
  auto dtype = input.scalar_type();

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& roundingMeta = meta.emplace_back();
  roundingMeta.guid = c10::isIntegralType(dtype, true) ? "identity" : guid;

  SharedMetaTensor inOutTensor = {rank, dtype};
  roundingMeta.inputs_data = {inOutTensor};
  roundingMeta.outputs_data = {inOutTensor};
  return meta;
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
          if (isI32Unsupported) {
            dtype = torch::kFloat32;
          } else {
            dtype = isDtypeUnsupported ? torch::kInt32 : dtype;
          }
        } else if (isDtypeUnsupported) {
          dtype = torch::kFloat32;
        }
      }

      auto& foreachMeta = metaVec.emplace_back(guid);
      foreachMeta.inputs_data = {{rank, dtype}};
      foreachMeta.outputs_data = {{rank, dtype}};
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
      std::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  if ((guid == "less" || guid == "less_fwd") &&
      inputType == c10::ScalarType::Short) {
    inputType = c10::ScalarType::Int;
  }

  SharedMetaDataVector meta;
  meta.reserve(2); // Reserve +1 for cases it is used somewhere else
  auto& compareSharedMeta = meta.emplace_back(guid);
  compareSharedMeta.inputs_data = {
      {selfRank, inputType}, {otherRank, inputType}};
  compareSharedMeta.outputs_data = {{outputRank, c10::ScalarType::Bool}};
  return meta;
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
  SharedMetaDataVector metaVec;
  metaVec.reserve(selfsSize * 3);
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
    const auto outputRank = std::max({selfRank, tensor1Rank, tensor2Rank});
    bool isAddcdiv = guid == "addcdiv_fwd";
    const bool isOutputIntegral = c10::isIntegralType(dtype, true);
    dtype = (isAddcdiv && isOutputIntegral) ? torch::kFloat32 : dtype;
    bool hasFloor = false;

    if (isValueTensor) {
      const auto& valueTensor = value.toTensor();
      auto valueRank = valueTensor.dim();
      auto valueDtype = valueTensor.scalar_type();
      auto& sliceAxisSharedMeta = metaVec.emplace_back("slice_axis");
      sliceAxisSharedMeta.inputs_data.emplace_back(valueRank, valueDtype);
      sliceAxisSharedMeta.outputs_data.emplace_back(1, valueDtype);

      if (c10::isFloatingType(valueDtype) && isOutputIntegral) {
        auto& floorSharedMeta = metaVec.emplace_back("floor_fwd");
        hasFloor = true;
        floorSharedMeta.inputs_data = sliceAxisSharedMeta.outputs_data;
        floorSharedMeta.outputs_data = sliceAxisSharedMeta.inputs_data;
      }
    }

    auto& compositeSharedMeta = metaVec.emplace_back(guid);
    compositeSharedMeta.inputs_data = {
        {selfRank, dtype}, {tensor1Rank, dtype}, {tensor2Rank, dtype}};
    if (hasFloor) {
      compositeSharedMeta.inputs_data.emplace_back(1, dtype);
    }
    compositeSharedMeta.outputs_data.emplace_back(outputRank, dtype);
  }
  return metaVec;
}

SharedMetaDataVector BoolCastSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  auto input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();
  SharedMetaDataVector metaVec;
  // This sometimes is used with combination of other operator, reserve
  // additional space for it.
  metaVec.reserve(3);

  auto& equalFwdMeta = metaVec.emplace_back("equal_fwd");
  equalFwdMeta.inputs_data = {{rank, dtype}, {rank, dtype}};
  equalFwdMeta.outputs_data = {{rank, at::kBool}};

  auto& notFwdMeta = metaVec.emplace_back("not_fwd");
  notFwdMeta.inputs_data = equalFwdMeta.outputs_data;
  notFwdMeta.outputs_data = equalFwdMeta.outputs_data;
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
    } else if ((guid == "or" || guid == "xor") && !isI16orI64) {
      if (!(isSelfIntegral || isOtherIntegral) ||
          (((isSelfIntegral ^ isOtherIntegral) != 0) &&
           ((c10::elementSize(selfDtype) == 1 ||
             c10::elementSize(otherDtype) == 1)))) {
        promoteToCommonType = true;
      }
    }
  }

  if (promoteToCommonType) {
    auto computeDtype = habana_helpers::DTypeHelper::get_compute_dtype(
        {self, other},
        std::nullopt,
        habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
        false,
        std::nullopt,
        false,
        false);
    selfDtype = computeDtype;
    otherDtype = computeDtype;
  }

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& logicalBinaryMeta = meta.emplace_back(guid);
  logicalBinaryMeta.inputs_data = {
      {selfRank, selfDtype}, {other.dim(), otherDtype}};
  logicalBinaryMeta.outputs_data = {{selfRank, at::kBool}};
  return meta;
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
  } else {
    metaVec.reserve(1);
  }

  if (c10::isIntegralType(inputDtype, true)) {
    inputDtype = c10::ScalarType::Int;
  }
  auto outputDtype = inputDtype;

  auto outputRank = rank;
  if (!keepDim) {
    int64_t dimNum = 0;
    if (stack.size() >= 2) {
      const auto& dim = stack.at(1);
      if (dim.isIntList()) {
        dimNum =
            dim.isNone() ? 0 : static_cast<int64_t>(dim.toIntVector().size());
      } else if (dim.isInt()) {
        dimNum = 1;
      }
    }
    outputRank = dimNum == 0 || outputRank == 0 ? 0 : outputRank - dimNum;
  }

  auto& reduceMaxMultiDimFwdMeta = metaVec.emplace_back(guid);
  reduceMaxMultiDimFwdMeta.options.allow_long_type = true;
  reduceMaxMultiDimFwdMeta.inputs_data = {{rank, inputDtype}};
  reduceMaxMultiDimFwdMeta.outputs_data.emplace_back(outputRank, outputDtype);
  reduceMaxMultiDimFwdMeta.outputs_data.emplace_back(
      outputRank, c10::ScalarType::Long);
  return metaVec;
}

SharedMetaDataVector AddInplaceSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto& other = stack.at(1);
  const auto dtype = self.scalar_type();
  const auto selfRank = self.dim();
  const auto otherRank = other.isTensor() ? other.toTensor().dim() : 1;
  const auto outputRank = std::max(selfRank, otherRank);

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& addSharedMeta = meta.emplace_back("add_fwd");
  addSharedMeta.inputs_data = {{selfRank, dtype}, {otherRank, dtype}};
  addSharedMeta.outputs_data = {{outputRank, dtype}};

  return meta;
}

SharedMetaDataVector BinaryWithAlphaSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& self = stack.at(0);
  const auto& other = stack.at(1);
  const auto& selfTensor = self.toTensor();
  const auto& selfRank = selfTensor.dim();
  const auto& otherRank = other.isTensor() ? other.toTensor().dim() : 1;
  const auto& outputRank = std::max(selfRank, otherRank);

  at::ScalarType outputType;
  if (other.isTensor()) {
    const at::Tensor& otherTensor = other.toTensor();

    if (!otherTensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      outputType = at::result_type(selfTensor, otherTensor);
    } else {
      // create a new dummy scalar with default type, and
      // then call result_type(Tensor, Scalar) variant
      at::Scalar newOtherScalar;
      if (at::is_floating_point(otherTensor)) {
        newOtherScalar = at::Scalar((float)1.0);
      } else {
        newOtherScalar = at::Scalar((int64_t)1);
      }
      outputType = at::result_type(selfTensor, newOtherScalar);
    }
  } else {
    const auto& otherScalar = other.toScalar();
    outputType = at::result_type(selfTensor, otherScalar);
  }

  const auto& alpha = stack.at(2).toScalar();
  SharedMetaDataVector metaVec;
  metaVec.reserve(2);
  if (alpha.equal(1)) {
    // This node will only appear in eager mode but there is no way to
    // distinguish mode here so both possibilities should be added to
    // verification
    auto& binaryKernelMeta = metaVec.emplace_back();
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
  }
  auto& binaryRef = metaVec.emplace_back("binary_with_alpha_fwd");
  binaryRef.inputs_data = {{selfRank, outputType}, {otherRank, outputType}};
  binaryRef.outputs_data = {{outputRank, outputType}};
  return metaVec;
}

SharedMetaDataVector BitwiseLogicalSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto self = stack.at(0);
  const auto& other = stack.at(1);
  auto dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      {self, self},
      std::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  auto inputRank = self.isTensor() ? self.toTensor().dim() : 1;
  auto otherRank = other.isTensor() ? other.toTensor().dim() : 1;
  auto outputRank = std::max(inputRank, otherRank);

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& bitwiseSharedMeta = meta.emplace_back(guid);
  bitwiseSharedMeta.inputs_data = {{inputRank, dtype}, {otherRank, dtype}};
  bitwiseSharedMeta.outputs_data.emplace_back(outputRank, dtype);
  return meta;
}

SharedMetaDataVector TopkSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  auto self = stack.at(0).toTensor();

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& topkMeta = meta.emplace_back("topk");
  topkMeta.inputs_data.emplace_back(self.dim(), self.scalar_type());
  topkMeta.outputs_data.emplace_back(self.dim(), self.scalar_type());
  topkMeta.outputs_data.emplace_back(self.dim(), c10::ScalarType::Int);

  return meta;
}

SharedMetaDataVector RandomSeedTensorInputSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto self = stack_tensor(stack, 0);
  const auto& seed = stack.at(3);
  SharedMetaTensor seedSharedTensor = {1, c10::ScalarType::Int};
  if (seed.isTensor()) {
    const auto& seedTensor = seed.toTensor();
    seedSharedTensor = {seedTensor.dim(), seedTensor.scalar_type()};
  }

  const auto isUniform =
      guid.find("philox_random_uniform") != std::string::npos;
  auto computeDtype = self.scalar_type();
  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& randomSharedMeta = meta.emplace_back(guid);
  if (!isUniform) {
    // SL isn't able to correctly determine precision type from the first input.
    // For types half (log_normal_fwd kernel only) and f8 there is no dedicated
    // kernel and in such case fallback is forced. After fallback there will be
    // upcast fo f32 and op will be handled correctly.
    if ((guid.find("log_normal_fwd") != std::string::npos &&
         (computeDtype == c10::ScalarType::Half ||
          computeDtype == c10::ScalarType::Float8_e4m3fn ||
          computeDtype == c10::ScalarType::Float8_e5m2)) ||
        (guid.find("random_normal_fwd") != std::string::npos &&
         (computeDtype == c10::ScalarType::Float8_e4m3fn ||
          computeDtype == c10::ScalarType::Float8_e5m2))) {
      randomSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Int);
    } else {
      randomSharedMeta.inputs_data.push_back(
          createOptionalNotPresentSharedMetaTensor());
    }
    if (computeDtype != c10::ScalarType::BFloat16) {
      computeDtype = c10::ScalarType::Float;
    }
  }

  randomSharedMeta.inputs_data.push_back(seedSharedTensor);
  if (isUniform) {
    randomSharedMeta.inputs_data.push_back(randomSharedMeta.inputs_data[0]);
  }
  randomSharedMeta.outputs_data.emplace_back(self.dim(), computeDtype);
  return meta;
}

SharedMetaDataVector PadBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  auto grad = stack_tensor(stack, 0);
  auto self = stack_tensor(stack, 1);
  auto dtype = self.scalar_type();

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& padBwdSharedMeta = meta.emplace_back("pad_bwd");
  padBwdSharedMeta.inputs_data.emplace_back(grad.dim(), dtype);
  padBwdSharedMeta.outputs_data.emplace_back(self.dim(), dtype);
  return meta;
}

SharedMetaDataVector MatrixMulWithAddSharedMeta(
    const at::Stack& stack,
    const std::string& guid,
    bool activation_variant) {
  const auto& input = stack_tensor(stack, 0);
  const auto& mat1 = stack_tensor(stack, 1);
  const auto& mat2 = stack_tensor(stack, 2);
  const bool isAddMM = guid == "addmm" || guid == "addbmm" || guid == "baddbmm";
  auto outputRank = 1;
  if (isAddMM) {
    if (guid == "baddbmm") {
      outputRank = 3;
    } else {
      outputRank = 2;
    }
  }
  const auto precisionType = mat1.scalar_type();

  SharedMetaDataVector metaVec;
  metaVec.reserve(2);
  auto& matrixMulSharedMeta = metaVec.emplace_back(guid);
  matrixMulSharedMeta.inputs_data = {
      {input.dim(), precisionType},
      {mat1.dim(), precisionType},
      {mat2.dim(), precisionType}};
  matrixMulSharedMeta.outputs_data.emplace_back(outputRank, precisionType);

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();
  const bool shouldUseParams = beta_val == 0.0 || beta_val == 1.0 ||
      alpha_val == 1.0 || (isAddMM && alpha_val == 0.0);
  if (shouldUseParams) {
    matrixMulSharedMeta.inputs_data.emplace_back(1, precisionType);
    matrixMulSharedMeta.inputs_data.emplace_back(1, precisionType);
  }

  const bool append_activation = alpha_val != 0 || beta_val != 0;
  if (append_activation && activation_variant) {
    const bool use_gelu = stack.at(5).toBool();

    auto& geluReluSharedMeta =
        metaVec.emplace_back(use_gelu ? "gelu_fwd" : "relu_fwd");
    geluReluSharedMeta.inputs_data = matrixMulSharedMeta.outputs_data;
    geluReluSharedMeta.outputs_data = geluReluSharedMeta.inputs_data;

    if (use_gelu) {
      geluReluSharedMeta.outputs_data.push_back(
          geluReluSharedMeta.outputs_data[0]);
    }
  }
  return metaVec;
}

SharedMetaDataVector MaxPoolWithIndicesFwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& self = stack_tensor(stack, 0);
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();
  auto indexType = c10::ScalarType::Long;

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& maxPoolWithIndicesSharedMeta = meta.emplace_back(guid);
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
  } else {
    maxPoolWithIndicesSharedMeta.options.allow_long_type = true;
  }
  maxPoolWithIndicesSharedMeta.outputs_data = {
      {rank, indexType}, {rank, dtype}};

  return meta;
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
  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& maxPoolWithIndicesSharedMeta = meta.emplace_back(guid);
  maxPoolWithIndicesSharedMeta.inputs_data.emplace_back(
      grad.dim(), grad.scalar_type());
  maxPoolWithIndicesSharedMeta.inputs_data.emplace_back(rank, dtype);
  maxPoolWithIndicesSharedMeta.inputs_data.emplace_back(
      indices.dim(), indexType);
  maxPoolWithIndicesSharedMeta.options.allow_long_type = true;
  maxPoolWithIndicesSharedMeta.outputs_data = {{rank, dtype}};

  return meta;
}

SharedMetaDataVector EmptySharedMeta(
    const at::Stack& /*unused*/,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  // op doesn't call any kernels or [SW-205149] return empty vector because
  // shape tensor validation will block shape agnostic flow
  return {};
}

SharedMetaDataVector MatmulSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
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
    if (matmul3d2dReshapeEnabled && selfRank == 3 && otherRank == 2) {
      addBias = isBiasPresentForBmm;
    }
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

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& gemmSharedMeta = meta.emplace_back(guid);
  gemmSharedMeta.inputs_data = {{selfRank, dtype}, {otherRank, dtype}};
  if (addBias) {
    const auto& bias = stack_tensor(stack, 2);
    gemmSharedMeta.inputs_data.emplace_back(bias.dim(), bias.scalar_type());
  }
  gemmSharedMeta.outputs_data.emplace_back(outputRank, dtype);

  return meta;
}

SharedMetaDataVector StridedViewCommonSharedMeta(
    const int64_t inputDim,
    const int64_t outputDim,
    const c10::ScalarType dtype) {
  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& stridedViewSharedMeta = meta.emplace_back("strided_view");
  stridedViewSharedMeta.inputs_data.emplace_back(inputDim, dtype);
  stridedViewSharedMeta.outputs_data.emplace_back(outputDim, dtype);

  return meta;
}

SharedMetaDataVector StridedViewSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto dtype = self.scalar_type();
  const auto& sizes = stack.at(1);

  int64_t outputRank;
  if (sizes.isTensor()) {
    const auto& sizesTensor = stack_tensor(stack, 1);
    outputRank = sizesTensor.dim();
  } else {
    outputRank = static_cast<int64_t>(sizes.toListRef().size());
  }

  return StridedViewCommonSharedMeta(self.dim(), outputRank, dtype);
}

SharedMetaDataVector AliasSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto dtype = self.scalar_type();

  return StridedViewCommonSharedMeta(self.dim(), self.dim(), dtype);
}

SharedMetaDataVector InstanceNormSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  auto rank = self.dim() > 3 ? self.dim() : 4;
  const auto dtype = self.scalar_type();

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& instanceNormSharedMeta = meta.emplace_back("instance_norm_fwd");
  instanceNormSharedMeta.inputs_data = {
      {rank, dtype}, {1, c10::ScalarType::Float}, {1, c10::ScalarType::Float}};
  instanceNormSharedMeta.outputs_data = {
      {rank, dtype}, {2, c10::ScalarType::Float}, {2, c10::ScalarType::Float}};
  return meta;
}

SharedMetaDataVector KlDivSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto& target = stack_tensor(stack, 1);
  const auto reduction = stack.at(2).toInt();
  const auto logTarget = stack.at(3).toBool();
  const auto dtype = self.scalar_type();
  const auto targetRank = target.dim();
  const auto selfRank = self.dim();
  const SharedMetaTensor targetTensor = {targetRank, dtype};
  const SharedMetaTensor selfTensor = {selfRank, dtype};
  SharedMetaDataVector metaVec;
  metaVec.reserve(5);
  if (logTarget) {
    auto& expSharedMeta = metaVec.emplace_back("exp_fwd");
    expSharedMeta.inputs_data = {targetTensor};
    expSharedMeta.outputs_data = {targetTensor};
  } else {
    auto& logSharedMeta = metaVec.emplace_back("log_fwd");
    logSharedMeta.inputs_data = {targetTensor};
    logSharedMeta.outputs_data = {targetTensor};

    auto& reluSharedMeta = metaVec.emplace_back("relu_bwd");
    reluSharedMeta.inputs_data = {targetTensor, targetTensor};
    reluSharedMeta.outputs_data = {targetTensor};
  }

  auto& subSharedMeta = metaVec.emplace_back("sub_fwd");
  subSharedMeta.inputs_data = {targetTensor, selfTensor};
  subSharedMeta.outputs_data.emplace_back(
      std::max(selfRank, targetRank), dtype);

  auto& mulSharedMeta = metaVec.emplace_back("mult");
  mulSharedMeta.inputs_data = {targetTensor, subSharedMeta.outputs_data[0]};
  mulSharedMeta.outputs_data = subSharedMeta.outputs_data;

  if (reduction != at::Reduction::Reduction::None) {
    const std::string reduceGuid = reduction == at::Reduction::Reduction::Sum
        ? "reduce_sum_fwd"
        : "reduce_mean_fwd";
    auto& reduceSharedMeta = metaVec.emplace_back(reduceGuid);
    reduceSharedMeta.inputs_data = mulSharedMeta.outputs_data;
    reduceSharedMeta.outputs_data = {{1, dtype}};
  }

  return metaVec;
}

SharedMetaDataVector CopySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto dtype = self.scalar_type();
  const auto rank = self.dim();
  const auto& dst = stack.at(1);

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& copySharedMeta = meta.emplace_back("copy_fwd");
  copySharedMeta.inputs_data.emplace_back(rank, dtype);
  if (dst.isTensor()) {
    copySharedMeta.inputs_data.emplace_back(dst.toTensor().dim(), dtype);
  }
  copySharedMeta.outputs_data.emplace_back(rank, dtype);

  return meta;
}

SharedMetaDataVector OneHotSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto dtype = self.scalar_type();
  const auto rank = self.dim();

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& oneHotSharedMeta = meta.emplace_back("one_hot_fwd");
  oneHotSharedMeta.inputs_data.emplace_back(rank, dtype);
  oneHotSharedMeta.outputs_data.emplace_back(rank + 1, dtype);
  return meta;
}

SharedMetaDataVector CommonForeachCopyMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode,
    SharedMetaCreateFunction sharedMetaCreator) {
  SharedMetaDataVector metaVec;
  const auto& selfs = stack[0].toList();
  auto selfsSize = selfs.size();
  metaVec.reserve(selfsSize);
  const auto& others = stack[1].toList();
  for (size_t i = 0; i < selfsSize; i++) {
    at::Stack oneIterationStack = {selfs[i], others[i]};

    auto oneIterationSharedMeta =
        sharedMetaCreator(oneIterationStack, executionMode);
    metaVec.insert(
        std::end(metaVec),
        std::make_move_iterator(std::begin(oneIterationSharedMeta)),
        std::make_move_iterator(std::end(oneIterationSharedMeta)));
  }

  return metaVec;
}

SharedMetaDataVector ForeachCopySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaCreateFunction sharedMetaCreator =
      [](const at::Stack& stack,
         habana_helpers::HabanaExecutionMode executionMode) {
        return CopySharedMeta(stack, executionMode);
      };

  return CommonForeachCopyMeta(stack, executionMode, sharedMetaCreator);
}

} // namespace habana
