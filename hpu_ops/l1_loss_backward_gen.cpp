/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/version.h>
#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR < 13))
#include "generated/l1_loss_backward.h"

namespace habana {

sizes_vec L1LossBackwardOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);
  return {self.sizes().vec()};
}

void L1LossBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 1).sizes();
  auto reduction = stack.at(3).toInt();

  double norm_factor = (reduction == at::Reduction::Reduction::Mean)
      ? 1 / static_cast<double>(stack_tensor(stack, 1).numel())
      : 1;
  auto t_norm_factor =
      ConstantHelper(graph, norm_factor, ScalarType(), inputshape);
  auto t_diff = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1), syn_in(2)},
      {{inputshape, ScalarType()}});
  // Computes output, If t_diff > 0, output = 1.
  // if t_diff < 0, output = -1.
  // if t_diff == 0, output = 0.
  auto t_sign = BuildOp(
      graph,
      "sign_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_diff.at(0).get()},
      {{inputshape, ScalarType()}});

  auto t_mul = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), t_norm_factor.get()},
      {{inputshape, ScalarType()}});
  auto grad_in = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_mul.at(0).get(), t_sign.at(0).get()},
      {{inputshape, ScalarType(), 0}});
  syn_out(0) = std::move(grad_in.at(0));
}
} // namespace habana
#endif
