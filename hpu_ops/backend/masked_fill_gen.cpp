/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

OutputMetaDataVector MaskedFillMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto mask_shape = stack_tensor(stack, 1).sizes();

  OutputMetaData meta{};

  meta.dtype = self.scalar_type();
  meta.shape = at::infer_size(self.sizes(), mask_shape);

  return {meta};
}

bool MaskedFillSTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  static_cast<void>(outputs);
  static_cast<void>(inputs);
  return true;
}

void MaskedFill::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto value = stack.at(2);

  auto value_dtype = value.isScalar() ? value.toScalar().type()
                                      : value.toTensor().scalar_type();
  value_dtype = habana_helpers::getInternalDtype(value_dtype);
  auto self_dtype = habana_helpers::getInternalDtype(self.scalar_type());

  std::vector<synTensor> inputs = {syn_in(1), syn_in(2), syn_in(0)};
  std::unique_ptr<synapse_helpers::tensor> cast;

  if (value_dtype != self_dtype) {
    cast = std::make_unique<synapse_helpers::tensor>(OpBackend::BuildCast(
        this, graph, syn_in(2), {1}, value_dtype, self_dtype));
    inputs[1] = cast->get();
  }

  auto out_shape = MaskedFillMeta(stack)[0].shape;

  auto result =
      BuildOp(graph, guid_, std::move(inputs), {{out_shape, ScalarType(), 0}});

  syn_out(0) = std::move(result[0]);
}
} // namespace habana
