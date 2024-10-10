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

#include "hpu_ops/custom_softmax.h"

namespace habana {

CustomSoftmax::CustomSoftmax(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "custom_softmax_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {}

void CustomSoftmax::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "CustomSoftmax::AddNode");
  const auto self = getNextInput<TensorsPair>(stackGetter);
  const auto flavor = getNextInput<int>(stackGetter);

  ns_CustomSoftmax::Params params{};
  params.flavor = flavor;

  if (flavor == 0) {
    SetGuid(get_guid_with_precision("softmax_fwd", ScalarType()));
    // For flavor=0, the classic softmax_fwd is used. In theory
    // it should get ns_Softmax::Params with dim=0, but luckily it has the same
    // size and param type as ns_CustomSoftmax. To make code simpler, especially
    // that in the near future softmax_fwd will be used always,
    // I just pass ns_CustomSoftmax::Params with flavor=0.
  }

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       {self.syn_t},
       {{self.pt_t.sizes().vec(), ScalarType(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
}
} // namespace habana

static const auto& CustomSoftmaxKernelRegistry = habana::KernelRegistry().add(
    "hpu::custom_softmax",
    KERNEL_FN_GLOBAL(habana::CustomSoftmax));
