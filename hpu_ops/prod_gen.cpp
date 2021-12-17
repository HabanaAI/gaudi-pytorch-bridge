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

namespace habana {
sizes_vec ProdOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const bool keepdim = stack.at(2).toBool();
  std::vector<int64_t> shape{self.sizes().vec()};
  auto dim = stack.at(1).toInt();
  shape = ReduceOperator::compute_output_shape(self, dim, keepdim);
  return {shape};
}
void ProdOut::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto self = stack.at(0).toTensor();
  const bool keepdim = stack.at(2).toBool();
  auto dim_ = stack.at(1).toInt();
  auto shape = ProdOutputShape(stack)[0];
  std::vector<int64_t> outshape{self.sizes().vec()};
  size_t size = 0;
  const auto& params = FillArgMinMaxParams(stack, size);

  // keepdim is false reduce dim using reshape.
  // custom outshape give two diff shape but
  // reduce prod_fwd guid  requried same shape for Keepdim True/false
  if (!keepdim) {
    auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
    outshape[dim] = 1;
    auto reduce_prod = BuildOp(
        graph,
        "reduce_prod_fwd",
        {syn_in(0)},
        {{outshape, ScalarType()}},
        params.get(),
        size);
    auto reshape = BuildOp(
        graph,
        "reshape",
        {reduce_prod[0].get()},
        {{shape, ScalarType(), is_output_persistent_list[0], 0}});
    syn_out(0) = std::move(reshape[0]);

    // keepdim is true directly mapping to the tpc kernel.
  } else {
    auto reduce_prod = BuildOp(
        graph,
        "reduce_prod_fwd",
        {syn_in(0)},
        {{shape, ScalarType(), is_output_persistent_list[0], 0}},
        params.get(),
        size);
    syn_out(0) = std::move(reduce_prod[0]);
  }
}
} // namespace habana
