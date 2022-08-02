/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/gather.h"
#include "hpu_op_helper.h"

namespace habana {

sizes_vec GatherOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto index = stack.at(2).toTensor();
  auto dim_ = stack.at(1).toInt();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  std::vector<int64_t> shape = self.sizes().vec();
  if (shape.size()) {
    // for gather op, output size is same as index
    if (self.dim() == index.dim()) {
      shape = index.sizes().vec();
    } else {
      // for index_select and other index ops
      shape[dim] = index.numel();
    }
  }
  return {shape};
}

std::shared_ptr<void> FillGatherParams(const at::Stack& stack, size_t& size) {
  auto self = stack.at(0).toTensor();
  int dim_ = stack.at(1).toInt();
  auto dim = get_dim_in_tpc_order(dim_, self.dim());
  at::Tensor indices = stack.at(2).toTensor();
  if (self.dim() != indices.dim()) {
    PARAMS_STUB(ns_GatherKernel::Params);
    params->axis = dim;
    return params;
  }
  PARAMS_STUB(ns_GatherElementsKernel::Params);
  params->axis = dim;
  return params;
}
void GatherHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // output shape
  auto outshape = GatherOutputShape(stack)[0];
  at::Tensor self = stack.at(0).toTensor();
  at::Tensor indices = stack.at(2).toTensor();

  // Fill params for gather_fwd or gather_elements_fwd guid
  size_t size = 0;
  auto params = FillParams(stack, size);
  if (self.dim() != indices.dim()) {
    this->guid_ =
        "gather_fwd_" + habana_helpers::name_suffix_from_type(ScalarType());
  } else {
    this->guid_ = "gather_elements_fwd_" +
        habana_helpers::name_suffix_from_type(ScalarType());
  }

  auto result = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1)},
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(result[0]);
  // }
}

} // namespace habana
