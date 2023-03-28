/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/masked_fill.h"

namespace habana {

void MaskedFill::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto value = stack.at(2);

  auto value_dtype = value.isScalar()
      ? habana_helpers::getInternalDtype(value.toScalar().type())
      : value.toTensor().scalar_type();

  std::vector<synTensor> inputs = {syn_in(1), syn_in(2), syn_in(0)};
  std::unique_ptr<synapse_helpers::tensor> cast;

  if (value_dtype != self.scalar_type()) {
    cast = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
        this, graph, syn_in(2), {1}, value_dtype, self.scalar_type()));
    inputs[1] = cast->get();
  }

  auto result =
      BuildOp(graph, guid_, inputs, {{self.sizes(), ScalarType(), 0}});

  syn_out(0) = std::move(result[0]);
}
} // namespace habana
