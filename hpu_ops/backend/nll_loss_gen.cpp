/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "generated/backend/nll_loss2d_backward.h"
#include "generated/backend/nll_loss2d_forward.h"
#include "generated/backend/nll_loss_backward.h"
#include "generated/backend/nll_loss_forward.h"

namespace habana {

OutputMetaDataVector NllLossFwdMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& target = stack_tensor(stack, 1);
  int64_t reduction = stack.at(3).toInt();
  OutputMetaDataVector meta(2);
  for (int i = 0; i < 2; ++i) {
    meta.at(i).dtype = self.scalar_type();
    meta.at(i).shape = {};
  }
  if (reduction == at::Reduction::Reduction::None) {
    meta.at(0).shape = target.sizes().vec();
  }
  return meta;
}

OutputMetaDataVector NllLossBwdMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& target = stack_tensor(stack, 1);
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = target.sizes().vec();
  return {meta};
}

sizes_vec NllLossBwdShapeTnsrShape(const at::Stack& stack) {
  // Return correct Shape Tensor size
  const torch::Tensor& target = stack_tensor(stack, 2);
  const torch::Tensor& input = stack_tensor(stack, 1);
  auto target_vec = target.sizes().vec();
  target_vec.push_back(input.sizes().vec().at(1));
  return {target_vec};
}

static std::shared_ptr<void> FillNllLossParams(
    size_t& size,
    int64_t reduction,
    int64_t ignore_index) {
  PARAMS_STUB(ns_NLLLossKernel::ParamsOptionalIgnoreIndex);
  switch (reduction) {
    case at::Reduction::Reduction::None:
      params->mode = NLLLossMode_t::NLL_LOSS_MODE_NONE;
      break;
    case at::Reduction::Reduction::Mean:
      params->mode = NLLLossMode_t::NLL_LOSS_MODE_MEAN;
      break;
    case at::Reduction::Reduction::Sum:
      params->mode = NLLLossMode_t::NLL_LOSS_MODE_SUM;
      break;
    default:
      TORCH_CHECK(false, "Unsupported reduction in nll_loss: ", reduction);
  }
  params->ignoreIndexValue = ignore_index;
  return params;
}

std::shared_ptr<void> FillNllLossFwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto ignore = stack.at(3).toInt();
  auto reduction = stack.at(4).toInt();
  return FillNllLossParams(size, ignore, reduction);
}

std::shared_ptr<void> FillNllLossBwdParams(
    const at::Stack& stack,
    size_t& size) {
  auto ignore = stack.at(4).toInt();
  auto reduction = stack.at(5).toInt();
  return FillNllLossParams(size, ignore, reduction);
}
enum modes { Fwd2D, Bwd2D };

static std::vector<synapse_helpers::tensor> NllLoss(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const OutputMetaData& meta,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_index = c10::nullopt) {
  return OpBackend::BuildNode(
      op,
      graph,
      {op->GetGuid(),
       std::move(input),
       {{meta.shape, meta.dtype, final_index}},
       params.get(),
       size});
}

static std::vector<synapse_helpers::tensor> NllLossBwdFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const OutputMetaData& meta,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_index = c10::nullopt,
    at::IntArrayRef shapeTnsrSize = {}) {
  // This helper function is used only when weight is none
  op->CreateShapeTensorInput(graph, meta.dtype, shapeTnsrSize, input);
  return NllLoss(op, graph, input, meta, params, size, final_index);
}

static void DummyOutput(
    synapse_helpers::graph& graph,
    PytorchKernelContextPtr& p_context_,
    bool persistent,
    bool external) {
  p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
      p_context_->pt_outputs_.at(1), graph, persistent, external));
}

static std::vector<synapse_helpers::tensor> ReduceWeight(
    OpBackend* op,
    const OutputMetaData& meta,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input) {
  ns_Reduction::Params reduce_params{};
  reduce_params.reductionDimension = 0;
  return OpBackend::BuildNode(
      op,
      graph,
      {

          get_guid_with_precision("reduce_sum_fwd", meta.dtype),
          std::move(input),
          {{1, meta.dtype}},
          &reduce_params,
          sizeof(reduce_params)});
}

static std::vector<synapse_helpers::tensor> ComputeWeightsSum(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    const OutputMetaData& meta,
    std::vector<synTensor> inputs) {
  constexpr auto synTargetIdx = 0;
  constexpr auto synWeightIdx = 1;

  const auto target = stack_tensor(stack, 1);
  const auto weights = stack_tensor(stack, 2);

  const auto targetSizes = target.sizes().vec();
  const auto targetFlattenSize = std::accumulate(
      targetSizes.cbegin(), targetSizes.cend(), 1, std::multiplies<int>{});

  auto flattenTarget = OpBackend::BuildFlatten(
      op,
      graph,
      std::move(inputs[synTargetIdx]),
      {targetFlattenSize},
      target.scalar_type());

  ns_GatherElementsKernel::Params gatherParams{};
  gatherParams.axis = 0;

  auto targetMappedToWeights = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("gather_elements_fwd", weights.scalar_type()),
       std::vector<synTensor>{
           std::move(inputs[synWeightIdx]), std::move(flattenTarget.get())},
       {{targetFlattenSize, weights.scalar_type()}},
       &gatherParams,
       sizeof(gatherParams)});

  ns_Reduction::Params reduceParams{.reductionDimension = 0};

  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("reduce_sum_fwd", meta.dtype),
       std::vector<synTensor>{std::move(targetMappedToWeights[0].get())},
       {{1, meta.dtype}},
       &reduceParams,
       sizeof(reduceParams)});
}

void NllLoss2DFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (!isOutputInfMode()) {
    // remove total_weight from output as it is unsupported
    p_context_->syn_outputs_.pop_back();
    // dummy output in place of total_weight
    DummyOutput(
        graph,
        p_context_,
        IsOutputPersistent(1),
        GetOutputMetaData(1).external);
  }

  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto meta = OutputMeta(stack)[0];

  std::vector<synapse_helpers::tensor> nll_loss;
  int64_t reduction = stack.at(3).toInt();
  if (reduction != at::Reduction::Reduction::None) {
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
  }

  if (stack.at(2).isNone()) { // weight is none
    nll_loss =
        NllLoss(this, graph, {syn_in(0), syn_in(1)}, meta, params, size, 0);
  } else { // weight is not none
    auto weight_sum = (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0)
        ? ComputeWeightsSum(this, graph, stack, meta, {syn_in(1), syn_in(2)})
        : ReduceWeight(this, meta, graph, {syn_in(2)});

    nll_loss = NllLoss(
        this,
        graph,
        {syn_in(0), syn_in(1), syn_in(2), weight_sum[0].get()},
        meta,
        params,
        size,
        0);
  }

  syn_out(0) = std::move(nll_loss[0]);

  if (isOutputInfMode()) {
    auto self = stack_tensor(stack, 0);
    GetOutputInfMeta().AddOutputTensor(TensorMetaData(
        meta.shape,
        habana::HabanaOperator::CalculateStrides(
            meta.shape, self.suggest_memory_format()),
        self.scalar_type(),
        self.suggest_memory_format()));
  }
}

void NllLossBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto meta = OutputMeta(stack)[0];
  const auto shapeTnsrSize = NllLossBwdShapeTnsrShape(stack)[0];

  if (stack.at(3).isNone()) { // weight is none
    auto nll_loss = NllLossBwdFunc(
        this,
        graph,
        {syn_in(0), syn_in(2)},
        meta,
        params,
        size,
        0,
        shapeTnsrSize);
    syn_out(0) = std::move(nll_loss[0]);
  } else { // weight is not none
    auto nll_loss = BuildOp(
        graph,
        get_guid_with_precision("cnll_loss_bwd", meta.dtype),
        {syn_in(0), syn_in(1), syn_in(2), syn_in(3), syn_in(4)},
        {{meta.shape, meta.dtype, 0}},
        params.get(),
        size);
    syn_out(0) = std::move(nll_loss[0]);
  }
}

void NllLoss2DBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto meta = OutputMeta(stack)[0];
  const auto shapeTnsrSize = NllLossBwdShapeTnsrShape(stack)[0];

  // A JIRA is created for self input tensor not used
  // https://jira.habana-labs.com/browse/SW-73878

  std::vector<synapse_helpers::tensor> output;
  if (stack.at(3).isNone()) { // weight is none
    output = NllLossBwdFunc(
        this,
        graph,
        {syn_in(0), syn_in(2)},
        meta,
        params,
        size,
        0,
        shapeTnsrSize);
  } else { // weight is not none
    auto weight_sum = ReduceWeight(this, meta, graph, {syn_in(3)});

    output = NllLoss(
        this,
        graph,
        {syn_in(0), syn_in(2), syn_in(3), weight_sum[0].get()},
        meta,
        params,
        size,
        0);
  }
  syn_out(0) = std::move(output[0]);
}
} // namespace habana
