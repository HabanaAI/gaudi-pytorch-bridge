/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/nll_loss2d_backward.h"
#include "generated/nll_loss2d_forward.h"
#include "generated/nll_loss_backward.h"
#include "generated/nll_loss_forward.h"
#include "hpu_op_helper.h"

namespace habana {
sizes_vec NllLossFwdOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& target = stack_tensor(stack, 1);
  int64_t reduction = stack.at(3).toInt();
  if (reduction == at::Reduction::Reduction::None) {
    return {target.sizes().vec(), {}};
  }
  return {{}, {}};
}

sizes_vec NllLossBwdOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& target = stack_tensor(stack, 1);
  return {target.sizes().vec()};
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

// Transpose NCHW to NHWC and vice versa
static std::vector<synapse_helpers::tensor> Transpose_MemFormat(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes nll_loss_mode,
    std::vector<synTensor> input,
    const at::IntArrayRef input_shape,
    c10::optional<int> final_index = c10::nullopt) {
  synTransposeParams trans_params{};
  trans_params.tensorDim = 4;
  for (int i = 0; i < 4; ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  if (nll_loss_mode == Fwd2D) { // 2D variant Fwd
    std::swap(trans_params.permutation[1], trans_params.permutation[2]);
    std::swap(trans_params.permutation[0], trans_params.permutation[1]);
  } else if (nll_loss_mode == Bwd2D) { // 2D variant Bwd
    std::swap(trans_params.permutation[0], trans_params.permutation[1]);
    std::swap(trans_params.permutation[1], trans_params.permutation[2]);
  }
  return OpBackend::BuildNode(
      op,
      graph,
      {"transpose",
       std::move(input),
       {{input_shape, op->ScalarType(), final_index}},
       &trans_params,
       sizeof(trans_params)});
}

static std::vector<synapse_helpers::tensor> NllLoss(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_index = c10::nullopt) {
  return OpBackend::BuildNode(
      op,
      graph,
      {op->GetGuid(),
       std::move(input),
       {{outshape, op->ScalarType(), final_index}},
       params.get(),
       size});
}

static std::vector<synapse_helpers::tensor> NllLossBwdFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    c10::ScalarType dtype,
    const at::IntArrayRef outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_index = c10::nullopt) {
  // This helper function is used only when weight is none
  op->CreateShapeTensorInput(graph, dtype, outshape, input);
  return NllLoss(op, graph, input, outshape, params, size, final_index);
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
    synapse_helpers::graph& graph,
    std::vector<synTensor> input) {
  ns_Reduction::Params reduce_params{};
  reduce_params.reductionDimension = 0;
  return OpBackend::BuildNode(
      op,
      graph,
      {"reduce_sum_fwd_" +
           habana_helpers::name_suffix_from_type(op->ScalarType()),
       std::move(input),
       {{1, op->ScalarType()}},
       &reduce_params,
       sizeof(reduce_params)});
}

void NllLossFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // remove total_weight from output as it is unsupported
  // JIRA https://jira.habana-labs.com/browse/SW-73520
  p_context_->syn_outputs_.pop_back();
  // dummy output in place of total_weight
  DummyOutput(
      graph,
      p_context_,
      IsOutputPersistent(1),
      m_output_metadata.at(1).external);

  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto outshape = ComputeOutputShapes(stack, true)[0];

  if (stack.at(2).isNone()) { // weight is none
    auto nll_loss =
        NllLoss(this, graph, {syn_in(0), syn_in(1)}, outshape, params, size, 0);
    syn_out(0) = std::move(nll_loss[0]);
  } else { // weight is not none
    auto weight_sum = ReduceWeight(this, graph, {syn_in(2)});
    auto nll_loss = NllLoss(
        this,
        graph,
        {syn_in(0), syn_in(1), syn_in(2), weight_sum[0].get()},
        outshape,
        params,
        size,
        0);
    syn_out(0) = std::move(nll_loss[0]);
  }
}

void NllLoss2DFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // remove total_weight from output as it is unsupported
  p_context_->syn_outputs_.pop_back();
  // dummy output in place of total_weight
  DummyOutput(
      graph,
      p_context_,
      IsOutputPersistent(1),
      m_output_metadata.at(1).external);

  auto input_shape = stack_tensor(stack, 0).sizes();
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto outshape = ComputeOutputShapes(stack, true)[0];

  std::vector<synapse_helpers::tensor> nll_loss;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    int64_t reduction = stack.at(3).toInt();
    if (reduction != at::Reduction::Reduction::None) {
      kernel_meta_data_.synapse_output_layout.assign(
          {synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    }

    if (stack.at(2).isNone()) { // weight is none
      nll_loss = NllLoss(
          this, graph, {syn_in(0), syn_in(1)}, outshape, params, size, 0);
    } else { // weight is not none
      auto weight_sum = ReduceWeight(this, graph, {syn_in(2)});

      nll_loss = NllLoss(
          this,
          graph,
          {syn_in(0), syn_in(1), syn_in(2), weight_sum[0].get()},
          outshape,
          params,
          size,
          0);
    }
  } else {
    std::vector<int64_t> tranpose_shape = {
        input_shape[0], input_shape[2], input_shape[3], input_shape[1]};
    auto transpose =
        Transpose_MemFormat(this, graph, Fwd2D, {syn_in(0)}, tranpose_shape);

    if (stack.at(2).isNone()) { // weight is none
      nll_loss = NllLoss(
          this,
          graph,
          {transpose[0].get(), syn_in(1)},
          outshape,
          params,
          size,
          0);
    } else { // weight is not none
      auto weight_sum = ReduceWeight(this, graph, {syn_in(2)});

      nll_loss = NllLoss(
          this,
          graph,
          {transpose[0].get(), syn_in(1), syn_in(2), weight_sum[0].get()},
          outshape,
          params,
          size,
          0);
    }
  }
  syn_out(0) = std::move(nll_loss[0]);
}

void NllLossBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto outshape = ComputeOutputShapes(stack, true)[0];
  auto dtype = stack.at(0).toTensor().scalar_type();
  // A JIRA is created for self input tensor not used
  // https://jira.habana-labs.com/browse/SW-73878
  if (stack.at(3).isNone()) { // weight is none
    auto nll_loss = NllLossBwdFunc(
        this, graph, {syn_in(0), syn_in(2)}, dtype, outshape, params, size, 0);
    syn_out(0) = std::move(nll_loss[0]);
  } else { // weight is not none
    auto weight_sum = ReduceWeight(this, graph, {syn_in(3)});

    auto nll_loss = NllLoss(
        this,
        graph,
        {syn_in(0), syn_in(2), syn_in(3), weight_sum[0].get()},
        outshape,
        params,
        size,
        0);
    syn_out(0) = std::move(nll_loss[0]);
  }
}

void NllLoss2DBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto outshape = ComputeOutputShapes(stack, true)[0];
  auto dtype = stack.at(0).toTensor().scalar_type();

  // A JIRA is created for self input tensor not used
  // https://jira.habana-labs.com/browse/SW-73878

  std::vector<synapse_helpers::tensor> output;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    if (stack.at(3).isNone()) { // weight is none
      output = NllLossBwdFunc(
          this,
          graph,
          {syn_in(0), syn_in(2)},
          dtype,
          outshape,
          params,
          size,
          0);
    } else { // weight is not none
      auto weight_sum = ReduceWeight(this, graph, {syn_in(3)});

      output = NllLoss(
          this,
          graph,
          {syn_in(0), syn_in(2), syn_in(3), weight_sum[0].get()},
          outshape,
          params,
          size,
          0);
    }
  } else {
    auto input_shape = stack_tensor(stack, 1).sizes();
    std::vector<int64_t> loss_shape = {
        input_shape[0], input_shape[2], input_shape[3], input_shape[1]};
    if (stack.at(3).isNone()) { // weight is none
      auto nll_loss = NllLossBwdFunc(
          this, graph, {syn_in(0), syn_in(2)}, dtype, loss_shape, params, size);
      output = Transpose_MemFormat(
          this, graph, Bwd2D, {nll_loss[0].get()}, outshape, 0);
    } else { // weight is not none
      auto weight_sum = ReduceWeight(this, graph, {syn_in(3)});

      auto nll_loss = NllLoss(
          this,
          graph,
          {syn_in(0), syn_in(2), syn_in(3), weight_sum[0].get()},
          loss_shape,
          params,
          size);
      output = Transpose_MemFormat(
          this, graph, Bwd2D, {nll_loss[0].get()}, outshape, 0);
    }
  }
  syn_out(0) = std::move(output[0]);
}
} // namespace habana
