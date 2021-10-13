/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/hpu_op.h"

namespace habana {

// Copied from habana_kernels/threshold_kernels.cpp
void ThresholdBackwardHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  // TODO: Remove this base class once [SW-65399] is resolved
  TORCH_CHECK(
      stack.size() == 3,
      "Incorrect size of inputs expected for threshold operator");

  TORCH_CHECK(stack[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(stack[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(stack[2].isScalar(), "Input arg3 type expected to be scalar");

  auto threshold = stack[2].toScalar();

  TORCH_CHECK(
      threshold.to<float>() == 0.0,
      "Threshold values other than 0 are not supported")

  OpBackend::AddNode(graph, stack, is_output_persistent_list);
}

} // namespace habana
