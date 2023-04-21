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

#include "dropout.h"
#include "generated/backend/_fused_dropout.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
OutputMetaDataVector FusedDropoutMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta, meta};
}

void FusedDropout::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "FusedDropout::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto ratio = getNextInput<double>(stackGetter);
  auto seed = getNextInput<TensorsPair>(stackGetter);

  ns_DropoutKernel::Params params{};
  size_t paramsSize = sizeof(params);
  params.ratio = ratio;

  auto outputShape = FusedDropoutMeta(stack)[0].shape;

  auto dropout = BuildDropout(
      this,
      graph,
      {self, seed},
      {NodeAttr::NodeOutputAttr{outputShape, self.pt_t.scalar_type(), 0},
       NodeAttr::NodeOutputAttr{outputShape, at::kChar, 1}},
      &params,
      paramsSize);

  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
}
} // namespace habana
