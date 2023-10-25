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

#include "generated/backend/argmax.h"
#include "generated/backend/argmin.h"
#include "hpu_ops/backend/reduction_template.h"

namespace {

auto GetDimVector(const at::Stack& stack) {
  auto dim = stack.at(1);
  return dim.isNone() ? std::vector<int64_t>{}
                      : std::vector<int64_t>{dim.toInt()};
}

} // namespace
namespace habana {

OutputMetaDataVector ArgMinMaxMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const bool keepdim = stack.at(2).toBool();
  auto dimVector = GetDimVector(stack);

  OutputMetaData meta;
  meta.shape = ReductionOutputShape(self, dimVector, keepdim)[0];
  meta.dtype = c10::ScalarType::Long;
  return {meta};
}

void ArgMinMax::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  const bool keepdim = stack.at(2).toBool();

  auto meta = ArgMinMaxMeta(stack)[0];

  auto op = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      GetDimVector(stack),
      keepdim,
      guid_,
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana
