/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
  c10::optional<at::Tensor> output_tensor = c10::nullopt;
  c10::optional<c10::ScalarType> output_type = c10::nullopt;
  auto size = stack.size();
  if (size > 2 && stack.at(size - 1).isTensor()) {
    output_tensor = stack.at(size - 1).toTensor();
    output_type = output_tensor.value().scalar_type();
  }

  auto self = stack_tensor(stack, 0);
  auto other = stack_tensor(stack, 1);
  meta.shape = at::infer_size(self.sizes(), other.sizes());
  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      output_tensor,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      output_type);

  return {meta};
}

void XlogYOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = XlogYMeta(stack)[0];
  auto other = stack_tensor(stack, 1);
  auto logyShape = other.sizes().vec();
  auto logyDtype = other.scalar_type();

  auto logy = BuildOp(
      graph,
      get_guid_with_precision("log1p_fwd", logyDtype),
      {syn_in(1)},
      {{logyShape, logyDtype}});
  auto xlogy = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {syn_in(0), logy[0].get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(xlogy[0]);
}
} // namespace habana
