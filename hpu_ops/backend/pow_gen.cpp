/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "generated/backend/_foreach_pow.h"
#include "generated/backend/pow.h"
#include "hpu_ops/backend/foreach.h"

using namespace std::literals;
namespace habana {

std::shared_ptr<void> FillPowParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Power::Params);
  params->exp_val = stack.at(1).toScalar().toDouble();
  return params;
}

static synapse_helpers::tensor createForeachPowNode(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const std::vector<synTensor>& syn_inputs,
    const std::vector<at::IValue>& pt_inputs,
    int out_index) {
  if (pt_inputs[0].isTensor() && pt_inputs[1].isTensor()) {
    const at::Tensor& self = pt_inputs[0].toTensor();
    const at::Tensor& other = pt_inputs[1].toTensor();

    auto result_type = at::result_type(self, other);
    if (isIntegralType(result_type, true)) {
      result_type = torch::kFloat32;
    }

    auto outshape = at::infer_size(self.sizes(), other.sizes());
    return std::move(OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("pow_fwd"sv, result_type),
         syn_inputs,
         {{outshape, result_type, out_index}}})[0]);
  } else if (pt_inputs[0].isTensor() && pt_inputs[1].isScalar()) {
    const at::Tensor& self = pt_inputs[0].toTensor();
    const at::Scalar& other = pt_inputs[1].toScalar();

    auto result_type = at::result_type(self, other);
    if (isIntegralType(result_type, true)) {
      result_type = torch::kFloat32;
    }

    ns_Power::Params params{};
    params.exp_val = other.toDouble();

    return std::move(OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("pow_fwd"sv, result_type),
         syn_inputs,
         {{self.sizes().vec(), result_type, out_index}},
         &params,
         sizeof(params)})[0]);
  } else {
    const at::Scalar& self = pt_inputs[0].toScalar();
    const at::Tensor& other = pt_inputs[1].toTensor();

    auto result_type = at::result_type(self, other);
    if (isIntegralType(result_type, true)) {
      result_type = torch::kFloat32;
    }

    auto syn_self = OpBackend::BuildConstant(op, graph, self, result_type);
    return std::move(OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("pow_fwd"sv, result_type),
         {syn_self.get(), syn_inputs[0]},
         {{other.sizes().vec(), result_type, out_index}}})[0]);
  }
}

static SharedMetaDataVector ForeachPowOneIterationSharedMeta(
    const at::Stack& stack) {
  const auto& self = stack.at(0);
  const auto& other = stack.at(1);
  if (self.isTensor() && other.isTensor()) {
    const auto& selfTensor = self.toTensor();
    const auto& otherTensor = other.toTensor();
    auto selfRank = selfTensor.dim();
    auto otherRank = otherTensor.dim();
    auto outputRank = std::max(selfRank, otherRank);
    auto dtype = at::result_type(selfTensor, otherTensor);
    if (isIntegralType(dtype, true))
      dtype = torch::kFloat32;

    SharedMetaData powSharedMeta{"pow_fwd"};
    powSharedMeta.inputs_data = {{selfRank, dtype}, {otherRank, dtype}};
    powSharedMeta.outputs_data = {{outputRank, dtype}};
    return {powSharedMeta};
  } else if (self.isTensor() && other.isScalar()) {
    const auto& selfTensor = self.toTensor();
    auto rank = selfTensor.dim();
    const auto& otherScalar = other.toScalar();
    auto dtype = at::result_type(selfTensor, otherScalar);
    if (isIntegralType(dtype, true))
      dtype = torch::kFloat32;

    SharedMetaData powSharedMeta{"pow_fwd"};
    powSharedMeta.inputs_data = {{rank, dtype}, {1, dtype}};
    powSharedMeta.outputs_data = {{rank, dtype}};
    return {powSharedMeta};
  } else {
    const auto& selfScalar = self.toScalar();
    const auto& otherTensor = other.toTensor();
    auto rank = otherTensor.dim();
    auto dtype = at::result_type(selfScalar, otherTensor);
    if (isIntegralType(dtype, true))
      dtype = torch::kFloat32;

    SharedMetaData powSharedMeta{"pow_fwd"};
    powSharedMeta.inputs_data = {{1, dtype}, {rank, dtype}};
    powSharedMeta.outputs_data = {{rank, dtype}};
    return {powSharedMeta};
  }
}

SharedMetaDataVector PowForeachBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaCreateFunction sharedMetaCreator =
      [](const at::Stack& stack, habana_helpers::HabanaExecutionMode) {
        return ForeachPowOneIterationSharedMeta(stack);
      };

  return CommonForeachBinarySharedMeta(stack, executionMode, sharedMetaCreator);
}

SharedMetaDataVector PowBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return ForeachPowOneIterationSharedMeta(stack);
}

void PowForeachBinary::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  NodeCreateFunction node_creator = [](OpBackend* op,
                                       synapse_helpers::graph& graph,
                                       std::string&,
                                       const std::vector<synTensor>& syn_inputs,
                                       const std::vector<at::IValue>& pt_inputs,
                                       int out_index) {
    return createForeachPowNode(op, graph, syn_inputs, pt_inputs, out_index);
  };

  const size_t size = computeInputsNumber(stack);
  std::vector<synTensor> inputs(size);
  for (size_t i = 0; i < size; i++) {
    inputs[i] = syn_in(i);
  }
  auto results =
      CommonForeachBinary(this, guid_, inputs, graph, stack, node_creator);
  for (size_t i = 0; i < results.size(); i++) {
    syn_out(i) = std::move(results[i]);
  }
}

} // namespace habana
