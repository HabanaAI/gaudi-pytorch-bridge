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

sizes_vec GatherOutputShape(const at::Stack& stack, bool) {
  auto self = stack.at(0).toTensor();
  auto index = stack.at(2).toTensor();
  bool sparse_grad = stack.at(3).toBool();
  // Issue raised for sparse_grad support:
  // https://jira.habana-labs.com/browse/SW-67496
  TORCH_CHECK(
      sparse_grad == false, "Gather: spare_grad is not supported in TPC");

  TORCH_CHECK(
      self.dim() == index.dim(),
      "Gather: Expects input dimension and index dimension should be same but got ",
      "input dimension: ",
      self.dim(),
      "index dimension: ",
      index.dim());

  std::vector<int64_t> shape = index.sizes().vec();
  return {shape};
}

std::shared_ptr<void> FillGatherParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_GatherElementsKernel::Params);
  auto self = stack.at(0).toTensor();
  int dim = stack.at(1).toInt();
  params->axis = get_dim_in_tpc_order(dim, self.dim());
  return params;
}
} // namespace habana
