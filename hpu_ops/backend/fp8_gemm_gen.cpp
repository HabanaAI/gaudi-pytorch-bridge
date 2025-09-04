/**
 * Copyright (c) 2023-2025 Intel Corporation
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

#include "generated/backend/_fp8_gemm_bwd.h"
#include "generated/backend/fp8_gemm.h"
#include "generated/backend/fp8_gemm_v2.h"
#include "hpu_ops/common/batched_matmul_output_shape.h"
#include "hpu_ops/custom_op_outshape.h"
#include "hpu_ops/fp8_ops.h"
#include "hpu_ops/fp8_utils.h"

namespace sh = synapse_helpers;

namespace habana {
namespace {
using namespace habana::fp8;

void MaybeTransposeShape(std::vector<int64_t>& shape, bool isTranspose) {
  const auto rank = shape.size();
  if (isTranspose) {
    TORCH_CHECK(
        rank >= 2,
        "Input tensor must have at least 2 dimensions to perform transpose operation.");
    std::swap(shape[rank - 1], shape[rank - 2]);
  }
}

void HandleScale(
    habana::OpBackend* op,
    sh::graph& graph,
    const habana::VariantWrapper<TensorsPair, c10::IValue>& scaleOpt,
    const at::Tensor& input,
    bool isTranspose,
    std::vector<sh::tensor>& adjustedScale,
    std::vector<synTensor>& synInputs,
    int deviceId,
    const c10::IValue& scaleShape = c10::IValue{}) {
  if (scaleOpt.isTensorsPair()) {
    auto scale = scaleOpt.toTensorsPair();

    auto sizes = input.sizes().vec();
    MaybeTransposeShape(sizes, isTranspose);

    ValidateScaleShape(scale.pt_t, scaleShape);
    HandleScaleTensor(
        op,
        graph,
        scale.pt_t,
        scale.syn_t,
        adjustedScale,
        synInputs,
        scaleShape);
  } else {
    auto scale = scaleOpt.toIValue();
    ValidateScaleShape(scale, scaleShape);
    HandleScaleScalar(
        op, graph, scale, deviceId, adjustedScale, synInputs, scaleShape);
  }
}

void ValidateFp8GemmScales(const at::Stack& stack) {
  const auto& scaleA = stack[6];
  const auto& scaleB = stack[7];

  if (not(scaleA.isTensor() or scaleB.isTensor())) {
    return;
  }

  auto shapeA = stack_tensor(stack, 0).sizes().vec();
  const bool transA = stack[1].toBool();
  auto shapeB = stack_tensor(stack, 2).sizes().vec();
  const bool transB = stack[3].toBool();
  MaybeTransposeShape(shapeA, transA);
  MaybeTransposeShape(shapeB, transB);

  const std::vector<int64_t> shapeScaleA = scaleA.isTensor()
      ? scaleA.toTensor().sizes().vec()
      : std::vector<int64_t>{1};
  std::vector<int64_t> shapeScaleB = scaleB.isTensor()
      ? scaleB.toTensor().sizes().vec()
      : std::vector<int64_t>{1};
  const auto scaleBShape = stack[10];
  if (scaleBShape.isIntList() and scaleB.isTensor()) {
    const auto scaleShape = scaleBShape.toIntVector();
    const auto scaleShapeNumel = std::accumulate(
        scaleShape.begin(), scaleShape.end(), 1, std::multiplies<>{});
    TORCH_CHECK(
        scaleShapeNumel == scaleB.toTensor().numel(),
        "scaleShape input has different number of element than scaleB: ",
        scaleShapeNumel,
        " vs ",
        scaleB.toTensor().numel());
    shapeScaleB = scaleShape;
  }

  if (at::are_expandable(shapeA, shapeScaleA) and
      at::are_expandable(shapeB, shapeScaleB)) {
    return;
  }

  // Shapes are not broadcastable, so we're checking for per-block scaling
  // compatibility.
  const auto rankA = shapeA.size();
  const auto rankB = shapeB.size();
  const auto rankScaleA = shapeScaleA.size();
  const auto rankScaleB = shapeScaleB.size();
  TORCH_CHECK(
      rankA >= 2 and rankB >= 2,
      "Per-block scaling requires both inputs to be at least 2D.");

  const auto aDim0 = shapeA[rankA - 2];
  const auto aDim1 = shapeA[rankA - 1];
  const auto bDim0 = shapeB[rankB - 2];
  const auto bDim1 = shapeB[rankB - 1];
  const auto scaleADim0 = rankScaleA > 1 ? shapeScaleA[rankScaleA - 2] : 1;
  const auto scaleADim1 = shapeScaleA[rankScaleA - 1];
  const auto scaleBDim0 = rankScaleB > 1 ? shapeScaleB[rankScaleB - 2] : 1;
  const auto scaleBDim1 = shapeScaleB[rankScaleB - 1];

  TORCH_CHECK(
      aDim0 % scaleADim0 == 0 and aDim1 % scaleADim1 == 0,
      "For per-block scaling, the last two dimensions of the input A (",
      aDim0,
      ", ",
      aDim1,
      ") must be divisible by the last two dimensions of scale A (",
      scaleADim0,
      ", ",
      scaleADim1,
      ").");
  TORCH_CHECK(
      bDim0 % scaleBDim0 == 0 and bDim1 % scaleBDim1 == 0,
      "For per-block scaling, the last two dimensions of the input B (",
      bDim0,
      ", ",
      bDim1,
      ") must be divisible by the last two dimensions of scale B (",
      scaleBDim0,
      ", ",
      scaleBDim1,
      ").");

  const auto aRowBlockSize = aDim0 / scaleADim0;
  const auto aColBlockSize = aDim1 / scaleADim1;
  const auto bRowBlockSize = bDim0 / scaleBDim0;
  const auto bColBlockSize = bDim1 / scaleBDim1;

  TORCH_CHECK(
      aColBlockSize == bRowBlockSize and
          (aRowBlockSize == aColBlockSize or aRowBlockSize == 1) and
          (bColBlockSize == aColBlockSize or bColBlockSize == 1) and
          (aRowBlockSize * bColBlockSize != 1),
      "For per-block scaling, supported blocks configurations are: 1xN @ NxN, NxN @ Nx1, NxN @ NxN. Got: ",
      aRowBlockSize,
      "x",
      aColBlockSize,
      " @ ",
      bRowBlockSize,
      "x",
      bColBlockSize,
      ".");
}

synTensor GetSynTensorOrNullptr(
    const std::optional<TensorsPair>& tensorPairOpt) {
  return tensorPairOpt.has_value() ? tensorPairOpt->syn_t : nullptr;
}

} // namespace

/********** Fp8Gemm **********/

void Fp8Gemm::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "Fp8Gemm::AddNode");
  auto A = stackGetter.getNextInput<TensorsPair>();
  bool transA = stackGetter.getNextInput<bool>();
  auto B = stackGetter.getNextInput<TensorsPair>();
  bool transB = stackGetter.getNextInput<bool>();
  auto D = stackGetter.getNextInput<TensorsPair>();
  auto outType = stackGetter.getNextInput<c10::ScalarType>();
  auto scaleAOpt = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto scaleBOpt = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto biasOpt = stackGetter.getNextInput<std::optional<TensorsPair>>();
  bool accumulate = stackGetter.getNextInput<bool>();

  std::vector<int64_t> outShape;
  try {
    outShape =
        getBatchMatmulOutShape(A.pt_t.sizes(), B.pt_t.sizes(), transA, transB);
  } catch (const std::invalid_argument& e) {
    HABANA_ASSERT(false, e.what());
  }

  std::string guid = get_guid_with_precision("fp8_gemm"sv, outType);

  std::vector<synTensor> synInputs = {A.syn_t, B.syn_t};
  synInputs.push_back(GetSynTensorOrNullptr(scaleAOpt));
  synInputs.push_back(GetSynTensorOrNullptr(scaleBOpt));
  synInputs.push_back(GetSynTensorOrNullptr(biasOpt));
  if (accumulate) {
    synInputs.push_back(D.syn_t);
  }

  synGEMMParams params{transA, transB};

  auto gemm = OpBackend::BuildNode(
      this,
      graph,
      {guid, synInputs, {{outShape, outType, 0}}, &params, sizeof(params)});

  syn_out(0) = std::move(gemm[0]);
}

/********** Fp8GemmV2 **********/

// Left for now, used by torch.compile meta
sym_sizes_vec Fp8GemmOutShape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  HABANA_ASSERT(inputs.size() == 2);
  HABANA_ASSERT(params.size() == 2);
  return {getBatchMatmulOutShape(
      inputs[0].sym_sizes(),
      inputs[1].sym_sizes(),
      static_cast<bool>(params[0]),
      static_cast<bool>(params[1]))};
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(fp8_gemm, Fp8GemmOutShape);

OutputMetaDataVector Fp8GemmV2Meta(const at::Stack& stack) {
  auto A = stack_tensor(stack, 0);
  bool transA = stack[1].toBool();
  auto B = stack_tensor(stack, 2);
  bool transB = stack[3].toBool();
  OutputMetaData meta;
  try {
    meta.shape = getBatchMatmulOutShape(A.sizes(), B.sizes(), transA, transB);
  } catch (const std::invalid_argument& e) {
    HABANA_ASSERT(false, e.what());
    return {};
  }
  ValidateFp8GemmScales(stack);
  meta.dtype = stack[5].toScalarType();
  return {meta};
}

SharedMetaDataVector Fp8GemmSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const at::Tensor& A = stack_tensor(stack, 0);
  const at::Tensor& B = stack_tensor(stack, 2);

  SharedMetaData sharedMeta("fp8_gemm");
  // CGUID inputs order is: A, B, scales, bias, D
  sharedMeta.inputs_data.push_back(getSharedMetaFromTensor(A));
  sharedMeta.inputs_data.push_back(getSharedMetaFromTensor(B));
  sharedMeta.inputs_data.push_back(getSharedMetaTensorFromScale(stack.at(6)));
  sharedMeta.inputs_data.push_back(getSharedMetaTensorFromScale(stack.at(7)));
  sharedMeta.inputs_data.push_back(
      getSharedMetaFromOptionalTensor(stack.at(8).toOptional<at::Tensor>()));
  sharedMeta.inputs_data.push_back(
      getSharedMetaFromOptionalTensor(stack.at(4).toOptional<at::Tensor>()));

  sharedMeta.outputs_data.emplace_back(
      std::max(A.dim(), B.dim()), stack.at(5).toScalarType());

  return {sharedMeta};
}

void Fp8GemmV2::AddNode(sh::graph& graph, const at::Stack& stack) {
  HABANA_ASSERT(stack.size() == 11, "Fp8GemmV2 must have 11 input arguments");

  StackGetter stackGetter(this, stack, "Fp8Gemm::AddNode");
  auto A = stackGetter.getNextInput<TensorsPair>();
  bool transA = stackGetter.getNextInput<bool>();
  auto B = stackGetter.getNextInput<TensorsPair>();
  bool transB = stackGetter.getNextInput<bool>();
  auto DOpt = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto outType = stackGetter.getNextInput<c10::ScalarType>();
  auto scaleAOpt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto scaleBOpt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto biasOpt = stackGetter.getNextInput<std::optional<TensorsPair>>();
  bool accumulate = stackGetter.getNextInput<bool>();
  auto scaleShape = stackGetter.getNextInput<c10::IValue>();

  std::string guid = get_guid_with_precision("fp8_gemm"sv, outType);

  std::vector<synTensor> synInputs = {A.syn_t, B.syn_t};
  std::vector<sh::tensor> adjustedScale;

  HandleScale(
      this,
      graph,
      scaleAOpt,
      A.pt_t,
      transA,
      adjustedScale,
      synInputs,
      p_context_->device_id_);
  HandleScale(
      this,
      graph,
      scaleBOpt,
      B.pt_t,
      transB,
      adjustedScale,
      synInputs,
      p_context_->device_id_,
      scaleShape);

  synInputs.push_back(GetSynTensorOrNullptr(biasOpt));
  if (accumulate) {
    HABANA_ASSERT(
        DOpt,
        "Accumulation tensor must be provided at index 4 for Fp8GemmV2, when accumulate is true");
    synInputs.push_back(DOpt->syn_t);
  } else {
    synInputs.push_back(nullptr);
  }

  // GC pass FUSE_CONVERT_MME inserts this last input, but it needs
  // it to be explicitly filled with nullptr before.
  synInputs.push_back(nullptr);

  ns_Fp8Gemm::ParamsV2 params{};
  params.transpose_a = transA;
  params.transpose_b = transB;

  bool scaleANoneOrH2d = true;
  bool scaleBNoneOrH2d = true;
  if (scaleAOpt.isTensorsPair()) {
    const auto tmeta{get_tensor_extra_meta(scaleAOpt.toTensorsPair().pt_t)};
    scaleANoneOrH2d = tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR;
  }
  if (scaleBOpt.isTensorsPair()) {
    const auto tmeta{get_tensor_extra_meta(scaleBOpt.toTensorsPair().pt_t)};
    scaleBNoneOrH2d = tmeta->get_tensor_type() == HOST_TO_DEVICE_TENSOR;
  }

  if (scaleANoneOrH2d and scaleBNoneOrH2d) {
    const auto& device = habana::HPUDeviceContext::get_device();
    params.is_hw_aligned = device.get_scale_attribute_is_hw_aligned();
    params.scale_method_hash_id = device.get_scale_attribute_hash_id();
  }

  auto meta = Fp8GemmV2Meta(stack)[0];

  auto gemm = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       synInputs,
       {{meta.shape, meta.dtype, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(gemm[0]);
}

/********** Fp8GemmBwdV2 **********/

OutputMetaDataVector Fp8GemmBwdMeta(const at::Stack& stack) {
  const auto& gradIn = stack_tensor(stack, 0);
  const auto& A = stack_tensor(stack, 1);
  const auto& B = stack_tensor(stack, 3);

  OutputMetaDataVector meta = {
      getMetaFromTensor(A),
      getMetaFromTensor(B),
      getMetaFromTensor(gradIn),
      getMetaFromTensor(gradIn)};

  return meta;
}

SharedMetaDataVector Fp8GemmBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& gradIn = stack_tensor(stack, 0);
  const auto& A = stack_tensor(stack, 1);
  const auto& B = stack_tensor(stack, 3);
  const auto& AScale = stack.at(5);
  const auto& BScale = stack.at(6);

  const int gradInDim = gradIn.dim();
  const int ADim = A.dim();
  const int BDim = B.dim();

  const int AScaleDim = AScale.isTensor() ? AScale.toTensor().dim() : 0;
  const int BScaleDim = BScale.isTensor() ? BScale.toTensor().dim() : 0;

  const at::ScalarType gradInDtype = gradIn.scalar_type();
  const at::ScalarType ADtype = A.scalar_type();
  const at::ScalarType BDtype = B.scalar_type();

  const at::ScalarType AScaleDtype =
      AScale.isTensor() ? AScale.toTensor().scalar_type() : gradInDtype;
  const at::ScalarType BScaleDtype =
      BScale.isTensor() ? BScale.toTensor().scalar_type() : gradInDtype;

  SharedMetaData sharedMeta("fp8_gemm_bwd");
  sharedMeta.inputs_data.emplace_back(gradInDim, gradInDtype);
  sharedMeta.inputs_data.emplace_back(ADim, ADtype);
  sharedMeta.inputs_data.emplace_back(BDim, BDtype);
  sharedMeta.inputs_data.emplace_back(AScaleDim, AScaleDtype);
  sharedMeta.inputs_data.emplace_back(BScaleDim, BScaleDtype);

  sharedMeta.outputs_data.emplace_back(ADim, ADtype);
  sharedMeta.outputs_data.emplace_back(BDim, BDtype);
  sharedMeta.outputs_data.emplace_back(gradInDim, gradInDtype);
  sharedMeta.outputs_data.emplace_back(gradInDim, gradInDtype);

  return {sharedMeta};
}

void Fp8GemmBwd::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "Fp8Gemm::AddNode");
  auto gradIn = stackGetter.getNextInput<TensorsPair>();
  auto A = stackGetter.getNextInput<TensorsPair>();
  const bool transA = stackGetter.getNextInput<bool>();
  auto B = stackGetter.getNextInput<TensorsPair>();
  const bool transB = stackGetter.getNextInput<bool>();
  auto scaleAOpt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  auto scaleBOpt =
      stackGetter.getNextInput<std::variant<TensorsPair, c10::IValue>>();
  const bool hasBias = stackGetter.getNextInput<bool>();
  const bool hasAcc = stackGetter.getNextInput<bool>();

  std::string guid =
      get_guid_with_precision("fp8_gemm_bwd"sv, gradIn.pt_t.scalar_type());

  ns_Fp8Gemm::Params params{transA, transB};

  std::vector<synTensor> synInputs = {gradIn.syn_t, A.syn_t, B.syn_t};
  std::vector<sh::tensor> adjustedScale;

  HandleScale(
      this,
      graph,
      scaleAOpt,
      A.pt_t,
      transA,
      adjustedScale,
      synInputs,
      p_context_->device_id_);

  HandleScale(
      this,
      graph,
      scaleBOpt,
      B.pt_t,
      transB,
      adjustedScale,
      synInputs,
      p_context_->device_id_);

  auto meta = Fp8GemmBwdMeta(stack);
  std::vector<NodeAttr::NodeOutputAttr> outputAttrs = {
      {meta[0].shape, meta[0].dtype, 0}, {meta[1].shape, meta[1].dtype, 1}};

  if (hasBias) {
    outputAttrs.push_back({meta[2].shape, meta[2].dtype, 2});
  }
  if (hasAcc) {
    outputAttrs.push_back({meta[3].shape, meta[3].dtype, 3});
  }

  auto gemm = OpBackend::BuildNode(
      this,
      graph,
      {guid, synInputs, std::move(outputAttrs), &params, sizeof(params)});

  syn_out(0) = std::move(gemm[0]);
  syn_out(1) = std::move(gemm[1]);

  size_t outIdx = 2;
  if (hasBias) {
    syn_out(2) = std::move(gemm[outIdx++]);
  } else {
    AddUndefinedOutputTensor();
  }

  if (hasAcc) {
    syn_out(3) = std::move(gemm[outIdx]);
  } else {
    AddUndefinedOutputTensor();
  }
}

} // namespace habana
