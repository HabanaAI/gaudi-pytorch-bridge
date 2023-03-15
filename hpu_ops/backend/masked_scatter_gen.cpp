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

#include <string>
#include <vector>

#include "backend/synapse_helpers/habana_tensor.h"
#include "generated/backend/masked_scatter.h"
#include "hpu_ops/op_backend.h"

namespace habana {
void MaskedScatter::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "MaskedScatter::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto mask = getNextInput<TensorsPair>(stackGetter);
  auto source = getNextInput<TensorsPair>(stackGetter);

  auto selfShape = self.pt_t.sizes();
  auto maskScalarType = mask.pt_t.scalar_type();

  auto broadcastedMask =
      BroadcastHelper(graph, mask.syn_t, selfShape, maskScalarType);

  auto selfNrOfElements = self.pt_t.numel();

  auto nonZeroForBroadcastedMask = BuildNonZero(
      this,
      graph,
      broadcastedMask,
      {selfNrOfElements, self.pt_t.dim()},
      maskScalarType);

  auto flattenedSource = ReshapeHelper(
      graph, source.syn_t, {selfNrOfElements}, source.pt_t.scalar_type());

  constexpr auto finalResultIndex = 0;

  auto scatterND = BuildScatterNDOnnx(
      this,
      graph,
      {self.syn_t,
       nonZeroForBroadcastedMask[0].get(),
       flattenedSource.get(),
       nonZeroForBroadcastedMask[1].get()},
      selfShape,
      self.pt_t.scalar_type(),
      nonZeroForBroadcastedMask[1].shape().rank().value,
      finalResultIndex);

  syn_out(0) = std::move(scatterND);
}

} // namespace habana
