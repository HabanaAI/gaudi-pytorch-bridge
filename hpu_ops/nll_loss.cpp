/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

namespace habana {
sizes_vec HabanaOperatorHelper::NllLossOutputShape(
    const at::Stack& stack,
    bool) {
  const torch::Tensor& target = stack_tensor(stack, 1);
  int64_t reduction = stack.at(3).toInt();
  if (reduction == at::Reduction::Reduction::None) {
    return {target.sizes().vec(), {}};
  }
  return {{}, {}};
}

std::shared_ptr<void> HabanaOperatorHelper::FillNllLossParams(
    const at::Stack& stack,
    size_t& size) {
  int64_t reduction = stack.at(3).toInt();
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

  params->ignoreIndexValue = stack.at(4).toInt();
  return params;
}

void NllLoss::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  TORCH_CHECK(stack.at(2).isNone(), "NLL loss does not support weight.");

  // remove total_weight from output as it is unsupported
  p_context_->syn_outputs_.pop_back();

  HabanaOperatorHelper::AddNode(graph, stack, is_output_persistent_list);

  // dummy output in place of total_weight
  p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
      p_context_->pt_outputs_.at(1), graph, is_output_persistent_list[1]));
}
} // namespace habana
