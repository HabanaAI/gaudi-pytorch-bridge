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

#include "generated/backend/_fused_dropout.h"
#include "generated/backend/native_dropout.h"
#include "generated/backend/native_dropout_backward.h"
#include "hpu_ops/habana_random_ops.h"
#include "hpu_ops/op_backend.h"

namespace sh = synapse_helpers;

namespace habana {
using namespace std::literals;

std::vector<synapse_helpers::tensor> DropoutCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const FillParamsT& params,
    const OutputMetaDataVector& metas,
    std::vector<synTensor>& input_tensor) {
  auto dropout = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("dropout_fwd"sv, metas[0].dtype),
       input_tensor,
       {NodeAttr::NodeOutputAttr{metas[0].shape, metas[0].dtype, 0},
        NodeAttr::NodeOutputAttr{metas[1].shape, metas[1].dtype, 1}},
       params.ptr(),
       params.size()});
  return dropout;
}
FillParamsT FillFusedNativeDropoutParams(const at::Stack& stack) {
  PARAMS_STUB(ns_DropoutKernel::Params);
  const size_t ratioId =
      (stack.at(0).isTensor() && stack.at(1).isTensor()) ? 2 : 1;
  params->ratio = static_cast<float>(stack.at(ratioId).toScalar().toDouble());
  return paramsT;
}

OutputMetaDataVector FusedNativeDropoutMeta(const at::Stack& stack) {
  const size_t selfId =
      (stack.at(0).isTensor() && stack.at(1).isTensor()) ? 1 : 0;
  at::Tensor self = stack_tensor(stack, selfId);
  auto shape = self.sizes().vec();

  OutputMetaDataVector metas(2);
  metas[0].shape = shape;
  metas[0].dtype = self.scalar_type();
  metas[1].shape = shape;
  metas[1].dtype = at::kChar;

  return metas;
}

SharedMetaDataVector FusedNativeDropoutSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& seed = stack.at(2);
  auto isSeedTensor = seed.isTensor();
  at::ScalarType seedDtype = at::ScalarType::Int;
  int64_t seedRank = 1;
  if (isSeedTensor) {
    const auto& seedTensor = seed.toTensor();
    seedRank = seedTensor.dim();
    seedDtype = seedTensor.scalar_type();
  }

  auto self = (stack.at(0).isTensor() && stack.at(1).isTensor())
      ? stack_tensor(stack, 1)
      : stack_tensor(stack, 0);
  auto selfRank = self.dim();
  auto selfDtype = self.scalar_type();
  SharedMetaDataVector dropoutSharedMetaVec;
  dropoutSharedMetaVec.reserve(1);
  auto& dropoutSharedMeta = dropoutSharedMetaVec.emplace_back("dropout_fwd");
  dropoutSharedMeta.inputs_data = {
      {selfRank, selfDtype}, {seedRank, seedDtype}};
  dropoutSharedMeta.outputs_data = {
      {selfRank, selfDtype}, {selfRank, at::ScalarType::Char}};
  return dropoutSharedMetaVec;
}

void FusedNativeDropout::AddNode(sh::graph& graph, const at::Stack& stack) {
  const auto& seed = stack.at(2);
  auto params = FillParams(stack);
  auto metas = FusedNativeDropoutMeta(stack);

  std::vector<synTensor> inputTensors = {syn_in(0)};
  if (seed.isTensor()) {
    inputTensors.push_back(syn_in(1));
  } else {
    inputTensors.push_back(syn_seed());
  }
  auto dropout = DropoutCommon(this, graph, params, metas, inputTensors);
  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
}

SharedMetaDataVector NativeDropoutBackwardSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  // It is assumed that constant and cast kernels are handled for all
  // dtypes configuration, so shared layer omits validation.

  auto grad_output = stack_tensor(stack, 0);
  auto grad_dtype = grad_output.scalar_type();
  auto grad_rank = grad_output.dim();
  auto mask = stack_tensor(stack, 1);

  SharedMetaDataVector metaVec;
  metaVec.reserve(1);
  auto& dropoutSharedMeta = metaVec.emplace_back("dropout_bwd");
  dropoutSharedMeta.inputs_data = {
      {grad_rank, grad_dtype}, {mask.dim(), mask.scalar_type()}};
  dropoutSharedMeta.outputs_data = {{grad_rank, grad_dtype}};

  return metaVec;
}

FillParamsT FillNativeDropoutBackwardParams(const at::Stack& stack) {
  PARAMS_STUB(ns_DropoutKernel::Params);
  params->ratio = static_cast<float>(stack.at(2).toScalar().toDouble());
  return paramsT;
}

void NativeDropoutBackward::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "NativeDropoutBackward::AddNode");
  auto grad_output = stackGetter.getNextInput<TensorsPair>();
  auto grad_dtype = grad_output.pt_t.scalar_type();
  auto mask = stackGetter.getNextInput<TensorsPair>();
  auto params = FillParams(stack);

  auto dropout_bwd = BuildOp(
      graph,
      get_guid_with_precision("dropout_bwd"sv, grad_dtype),
      {grad_output.syn_t, mask.syn_t},
      {{grad_output.pt_t.sizes(), grad_dtype, 0}},
      &params,
      sizeof(params));

  syn_out(0) = std::move(dropout_bwd[0]);
}

//===----------------------------------------------------------------------===//
// This is the implementation of custom native dropout op in `torch.compile`
//===----------------------------------------------------------------------===//
HabanaNativeDropout::HabanaNativeDropout(
    int device_id,
    c10::ScalarType scalar_type)
    : HabanaRandomBase(device_id, "native_dropout", scalar_type, {1, 1}) {
  SetOutputMetaFn(FusedNativeDropoutMeta);
}

void HabanaNativeDropout::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto params = FillFusedNativeDropoutParams(stack);
  auto metas = FusedNativeDropoutMeta(stack);

  std::vector<synTensor> inputTensors = {syn_in(1), syn_in(0)};
  auto dropout = DropoutCommon(this, graph, params, metas, inputTensors);
  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
}

} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry()
        .REGISTER_HABANA_RANDOM_OP(native_dropout, NativeDropout)
        .REGISTER_HABANA_RANDOM_OP(_fused_dropout, NativeDropout);
