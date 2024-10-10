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

#include "generated/backend/logical_and.h"
#include "generated/backend/logical_not.h"
#include "generated/backend/logical_or.h"
#include "generated/backend/logical_xor.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

OutputMetaDataVector LogicalMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.shape = at::infer_size(
      stack.at(0).toTensor().sizes(), stack.at(1).toTensor().sizes());
  meta.dtype = at::kBool;

  return {meta};
}

OutputMetaDataVector LogicalNotMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.shape = stack.at(0).toTensor().sizes().vec();
  meta.dtype = at::kBool;

  return {meta};
}

SharedMetaDataVector LogicalNotSharedMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto rank = self.dim();
  auto dtype = self.scalar_type();
  SharedMetaDataVector metaVec = {};
  if ((self.scalar_type() == at::kFloat) ||
      (self.scalar_type() == at::kBFloat16) ||
      (self.scalar_type() == at::kInt) || (self.scalar_type() == at::kShort)) {
    metaVec = BoolCastSharedMeta({self});
    dtype = at::kBool;
  }

  SharedMetaData notMeta{"not"};
  notMeta.inputs_data = {{rank, dtype}};
  notMeta.outputs_data = {{rank, at::kBool}};
  metaVec.push_back(notMeta);
  return metaVec;
}

SharedMetaDataVector LogicalBinaryAndSharedMeta(const at::Stack& stack) {
  return LogicalBinarySharedMeta(stack, "and");
}

SharedMetaDataVector LogicalBinaryOrSharedMeta(const at::Stack& stack) {
  return LogicalBinarySharedMeta(stack, "or");
}

SharedMetaDataVector LogicalBinaryXorSharedMeta(const at::Stack& stack) {
  return LogicalBinarySharedMeta(stack, "xor");
}

void LogicalNotOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = LogicalNotMeta(stack).at(0);
  auto self = stack.at(0).toTensor();

  std::optional<synapse_helpers::tensor> castedInput{};
  if (self.scalar_type() == at::kFloat or self.scalar_type() == at::kBFloat16 or
      self.scalar_type() == at::kInt or self.scalar_type() == at::kShort) {
    castedInput =
        BuildBoolCast(this, graph, syn_in(0), self.sizes(), self.scalar_type());

    update_guid_dtype(guid_, at::kBool);
  }

  syn_out(0) = std::move(BuildOp(
      graph,
      guid_,
      {castedInput.has_value() ? castedInput.value().get() : syn_in(0)},
      {{meta.shape, meta.dtype, 0}})[0]);
}

} // namespace habana
