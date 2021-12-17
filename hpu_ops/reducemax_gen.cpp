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

void MaxOut::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();

  auto outshape = self.sizes().vec();
  auto shape = MinOutputShape(stack)[0];
  // Negative dimension
  dim = c10::maybe_wrap_dim(dim, self.dim(), true);
  outshape[dim] = 1;
  // FillMinParams, MinOutputShape does the same function as expected for this
  // op, so these functions are reused here
  size_t size = 0;
  const auto& params = FillMinParams(stack, size);
  auto dtype = c10::ScalarType::Int;

  if (!keepdim) {
    auto reduce_max = BuildOp(
        graph,
        guid_,
        {syn_in(0)},
        {{outshape, ScalarType()}, {outshape, dtype}},
        params.get(),
        size);

    auto reshape1 = BuildOp(
        graph,
        "reshape",
        {reduce_max[0].get()},
        {{shape, ScalarType(), is_output_persistent_list[0], 0}});

    auto reshape2 = BuildOp(
        graph,
        "reshape",
        {reduce_max[1].get()},
        {{shape, dtype, is_output_persistent_list[1], 1}});

    syn_out(0) = std::move(reshape1[0]);
    syn_out(1) = std::move(reshape2[0]);
  } else {
    auto reduce_max = BuildOp(
        graph,
        guid_,
        {syn_in(0)},
        {{outshape, ScalarType(), is_output_persistent_list[0], 0},
         {outshape, dtype, is_output_persistent_list[1], 1}},
        params.get(),
        size);

    syn_out(0) = std::move(reduce_max[0]);
    syn_out(1) = std::move(reduce_max[1]);
  }
}
} // namespace habana
