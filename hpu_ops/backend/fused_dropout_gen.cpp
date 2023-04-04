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

#include <ATen/Tensor.h>

namespace habana {
sizes_vec FusedDropoutOutputShape(const at::Stack& stack) {
  const auto& outputSizes = stack_tensor(stack, 0).sizes().vec();

  return {outputSizes, outputSizes};
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

  auto outputShapes = FusedDropoutOutputShape(stack);

  auto dropout = BuildDropout(
      this,
      graph,
      {self, seed},
      {NodeAttr::NodeOutputAttr{outputShapes[0], self.pt_t.scalar_type(), 0},
       NodeAttr::NodeOutputAttr{outputShapes[1], at::kChar, 1}},
      &params,
      paramsSize);

  syn_out(0) = std::move(dropout[0]);
  syn_out(1) = std::move(dropout[1]);
}
} // namespace habana
