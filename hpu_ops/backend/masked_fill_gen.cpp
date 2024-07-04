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

std::shared_ptr<void> FillMaskedFillParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_MaskedFill::Params);
  auto value = stack.at(2);
  if (value.isTensor()) {
    return params;
  }

  auto self = stack_tensor(stack, 0);
  auto self_dtype = habana_helpers::getInternalDtype(self.scalar_type());
  if (c10::isIntegralType(self_dtype, true)) {
    params->value.i = value.toScalar().toInt();
  } else {
    params->value.f = value.toScalar().toFloat();
  }

  return params;
}

void MaskedFill::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};

  auto out_shape = MaskedFillMeta(stack)[0].shape;
  const auto& params = FillMaskedFillParams(stack, size);

  auto value = stack.at(2);
  if (value.isTensor()) {
    inputs.push_back(syn_in(2));
  }

  auto result = BuildOp(
      graph,
      GetGuid(),
      std::move(inputs),
      {{out_shape, ScalarType(), 0}},
      params.get(),
      size);

  syn_out(0) = std::move(result[0]);
}

} // namespace habana
