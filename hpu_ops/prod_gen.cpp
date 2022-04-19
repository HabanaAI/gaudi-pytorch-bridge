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
#include "habana_kernels/reduction_kernels.h"
#include "reduction_op_util.h"

namespace habana {
sizes_vec ProdOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const bool keepdim = stack.at(2).toBool();
  std::vector<int64_t> shape{self.sizes().vec()};
  auto dim = stack.at(1).toInt();
  shape = ReduceOperator::compute_output_shape(self, dim, keepdim);
  return {shape};
}
void ProdOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  const bool keepdim = stack.at(2).toBool();
  auto dim_ = stack.at(1).toInt();
  auto shape = ProdOutputShape(stack)[0];

  auto reduce_prod = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      dim_,
      keepdim,
      "reduce_prod_fwd",
      {{shape, ScalarType(), 0}});
  syn_out(0) = std::move(reduce_prod[0]);
}
} // namespace habana
