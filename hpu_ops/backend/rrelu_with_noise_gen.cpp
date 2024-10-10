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
#include "generated/backend/rrelu_with_noise.h"
#include "generated/backend/rrelu_with_noise_backward.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

void Rrelu_with_noise::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "Rrelu_with_noise::AddNode");
  [[maybe_unused]] auto input = getNextInput<TensorsPair>(stackGetter);
  auto noiseIn = getNextInput<c10::optional<TensorsPair>>(stackGetter);
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
