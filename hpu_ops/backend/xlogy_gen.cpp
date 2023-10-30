/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/special_xlog1py.h"
#include "generated/backend/xlogy.h"

namespace habana {

OutputMetaDataVector XlogYMeta(const at::Stack& stack) {
  OutputMetaData meta;

  if (stack.at(1).isScalar()) {
    auto self = stack_tensor(stack, 0);
    meta.shape = self.sizes().vec();
    meta.dtype = self.scalar_type();
  } else if (stack.at(0).isScalar()) {
    auto other = stack_tensor(stack, 1);
    meta.shape = other.sizes().vec();
    meta.dtype = other.scalar_type();
  } else {
    auto self = stack_tensor(stack, 0);
    auto other = stack_tensor(stack, 1);
    meta.shape = at::infer_size(self.sizes(), other.sizes());
    meta.dtype = self.scalar_type();
  }

  return {meta};
}

void XlogYOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = XlogYMeta(stack)[0].shape;
  auto other_shape = stack_tensor(stack, 1).sizes().vec();

  auto logy = BuildOp(graph, guid_, {syn_in(1)}, {{other_shape, ScalarType()}});
  auto xlogy = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {syn_in(0), logy[0].get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(xlogy[0]);
}
} // namespace habana
