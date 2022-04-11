/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <bitset>
#include "generated/hpu_op.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_op_helper.h"
#include "reduction_op_util.h"
#define GUID "reduce_log_sum_exp_fwd_"

namespace habana {

sizes_vec LogSumExpOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> dim = stack.at(1).toIntList().vec();
  const bool keepdim = stack.at(2).toBool();
  std::vector<int64_t> compute_shape =
      ReduceOperator::compute_output_shape(self, dim, keepdim);
  return {compute_shape};
}

void LogSumExp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();

  const bool keepdim = stack.at(2).toBool();
  auto self_shape = self.sizes().vec();
  auto dim = stack.at(1).toIntVector();

  auto new_shape = ComputeOutputShapes(stack)[0];

  auto logsumexpout = HandleReductionDimAndKeepdim(
      this,
      graph,
      {syn_in(0)},
      dim,
      keepdim,
      GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      self_shape,
      new_shape,
      {{{}, ScalarType()}, {self_shape, ScalarType()}});

  syn_out(0) = std::move(logsumexpout.at(0));
}
} // namespace habana
