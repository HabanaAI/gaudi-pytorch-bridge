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

#include "hpu_ops/rms_norm.h"

namespace habana {

RMSNorm::RMSNorm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "rms_norm", scalar_type, {0, 0}, {}, {}, false) {}

void RMSNorm::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "RMSNorm::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto gamma = getNextInput<TensorsPair>(stackGetter);
  auto epsilon = getNextInput<double>(stackGetter);

  std::string guid =
      get_guid_with_precision("rms_norm_fwd", input.pt_t.scalar_type());

  ns_LayerNormKernel::Params params{};
  params.epsValid = true;
  params.eps = static_cast<float>(epsilon);

  std::vector<synTensor> inputs = {input.syn_t, gamma.syn_t};

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {input.pt_t.sizes(), input.pt_t.scalar_type(), 0},
      {input.pt_t.sizes(), c10::ScalarType::Float, 1}};

  auto output = OpBackend::BuildNode(
      this, graph, {guid, inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
  syn_out(1) = std::move(output[1]);
}

} // namespace habana

static const auto& RMSNormKernelRegistry = habana::KernelRegistry().add(
    "hpu::rms_norm",
    KERNEL_FN_GLOBAL(habana::RMSNorm));
