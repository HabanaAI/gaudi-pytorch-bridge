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
#include "generated/backend/rrelu_with_noise.h"
#include "generated/backend/rrelu_with_noise_backward.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

SharedMetaDataVector RreluWithNoiseSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();
  const auto training = stack.at(4).toBool();

  if (!training) {
    SharedMetaData leakyReluSharedMeta("leakyrelu_fwd");
    leakyReluSharedMeta.inputs_data.emplace_back(rank, dtype);
    leakyReluSharedMeta.outputs_data.emplace_back(rank, dtype);

    return {leakyReluSharedMeta};
  }

  SharedMetaDataVector out;
  SharedMetaData randUniformSharedMeta("random_uniform_fwd");
  randUniformSharedMeta.inputs_data.emplace_back(1, at::ScalarType::Int);
  randUniformSharedMeta.outputs_data.emplace_back(rank, dtype);
  out.push_back(randUniformSharedMeta);

  if (rank > 1) {
    SharedMetaData constantSharedMeta("constant");
    constantSharedMeta.outputs_data.emplace_back(rank, dtype);
    out.push_back(constantSharedMeta);
  }

  SharedMetaData lessEqSharedMeta("less_equal_fwd");
  lessEqSharedMeta.inputs_data.emplace_back(rank, dtype);
  lessEqSharedMeta.inputs_data.emplace_back(rank, dtype);
  lessEqSharedMeta.outputs_data.emplace_back(rank, at::ScalarType::Bool);
  out.push_back(lessEqSharedMeta);

  SharedMetaData whereSharedMeta("where_fwd");
  whereSharedMeta.inputs_data.emplace_back(rank, at::ScalarType::Bool);
  whereSharedMeta.inputs_data.emplace_back(rank, dtype);
  whereSharedMeta.inputs_data.emplace_back(rank, dtype);
  whereSharedMeta.outputs_data.emplace_back(rank, dtype);
  out.push_back(whereSharedMeta);

  SharedMetaData multSharedMeta("mult");
  multSharedMeta.inputs_data.emplace_back(rank, dtype);
  multSharedMeta.inputs_data.emplace_back(rank, dtype);
  multSharedMeta.outputs_data.emplace_back(rank, dtype);
  out.push_back(multSharedMeta);

  return out;
}

void Rrelu_with_noise::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "Rrelu_with_noise::AddNode");
  [[maybe_unused]] auto input = stackGetter.getNextInput<TensorsPair>();
  auto noiseIn = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto training = stack.at(4).toBool();
  auto lower = stack.at(2).toScalar().to<float>();
  auto upper = stack.at(3).toScalar().to<float>();
  size_t size = 0;
  if (training) {
    std::optional<synapse_helpers::tensor> noiseStorageOpt;
    auto [noise_in_storage_or_idx] = get_or_create_tensor<STORAGE_IDX>(
        *this,
        graph,
        noiseIn,
        (*noiseIn).pt_t.numel(),
        ScalarType(),
        0,
        noiseStorageOpt);
    PARAMS_STUB(ns_RandomUniform::Params);
    params->low = lower;
    params->high = upper;

    // populate seed
    std::vector<synTensor> inputs;
    if (stack.at(5).isTensor())
      inputs.push_back(syn_in(2));
    else
      inputs.push_back(syn_seed());
    CreateShapeTensorInput(graph, ScalarType(), outshape, inputs);

    // uniform random tensor
    auto uniform_random = BuildOp(
        graph,
        get_guid_with_precision("random_uniform_fwd", ScalarType()),
        std::move(inputs),
        {{outshape, ScalarType()}},
        params.get(),
        size);
    auto ones = ConstantHelper(graph, 1.0f, ScalarType(), outshape);
    auto zeros = ConstantHelper(graph, 0, ScalarType(), outshape);
    // cond: condition tensor
    auto cond = BuildOp(
        graph,
        get_guid_with_precision("less_equal_fwd", ScalarType()),
        {syn_in(0), zeros.get()},
        {{outshape, c10::ScalarType::Bool}});
    // noise: noise tensor
    auto noise = BuildOp(
        graph,
        get_guid_with_precision("where_fwd", ScalarType()),
        {cond[0].get(), uniform_random[0].get(), ones.get()},
        {NodeAttr::NodeOutputAttr{
            outshape,
            ScalarType(),
            c10::nullopt,
            DATA_TENSOR,
            syn_type_na,
            noise_in_storage_or_idx}});
    // output
    auto output = BuildOp(
        graph,
        get_guid_with_precision("mult", ScalarType()),
        {syn_in(0), noise[0].get()},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(output[0]);
    GetSynImplicitOutputs().emplace_back(PtInputIdxAndSynHelpTensor{
        1, std::move(noise[0]), std::get<int>(noise_in_storage_or_idx)});
  } else {
    PARAMS_STUB(ns_LeakyReluKernel::Params);
    auto negative_slope = (lower + upper) / 2;
    params->alpha = negative_slope;
    auto output = BuildOp(
        graph,
        get_guid_with_precision("leakyrelu_fwd", ScalarType()),
        {syn_in(0)},
        {{outshape, ScalarType(), 0}},
        params.get(),
        size);
    syn_out(0) = std::move(output[0]);
  }
}

SharedMetaDataVector RreluWithNoiseBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto grad = stack.at(0).toTensor();
  auto rank = grad.dim();
  auto resultType = grad.scalar_type();

  auto training = stack.at(5).toBool();
  auto lower = stack.at(3).toScalar().to<float>();
  auto upper = stack.at(4).toScalar().to<float>();
  auto guid = "leakyrelu_bwd";
  if (training && (upper - lower) > 1e-6) {
    guid = "mult";
  }
  SharedMetaData rreluBwdSharedMeta(guid);
  rreluBwdSharedMeta.inputs_data.emplace_back(rank, resultType);
  rreluBwdSharedMeta.inputs_data.emplace_back(rank, resultType);
  rreluBwdSharedMeta.outputs_data.emplace_back(rank, resultType);
  return {rreluBwdSharedMeta};
}

void Rrelu_with_noise_bwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto training = stack.at(5).toBool();
  auto lower = stack.at(3).toScalar().to<float>();
  auto upper = stack.at(4).toScalar().to<float>();
  if (training && (upper - lower) > 1e-6) {
    // grad_out * noise
    auto output = BuildOp(
        graph,
        get_guid_with_precision("mult", ScalarType()),
        {syn_in(0), syn_in(2)},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(output[0]);
  } else {
    size_t size = 0;
    PARAMS_STUB(ns_LeakyReluKernel::Params);
    auto negative_slope = (lower + upper) / 2;
    params->alpha = negative_slope;
    auto output = BuildOp(
        graph,
        get_guid_with_precision("leakyrelu_bwd", ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType(), 0}},
        params.get(),
        size);
    syn_out(0) = std::move(output[0]);
  }
}
} // namespace habana
