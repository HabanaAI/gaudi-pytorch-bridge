/*******************************************************************************
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
#include "generated/backend/gelu.h"
#include "generated/backend/gelu_backward.h"
#include "hpu_ops/op_backend.h"

namespace habana {
std::shared_ptr<void> FillGeluParams(
    const at::Stack& stack,
    size_t& size,
    int approx_index) {
  PARAMS_STUB(ns_GeluKernel::Params);
  if (GET_ENV_FLAG_NEW(PT_HPU_FORCE_TANH_FOR_GELU)) {
    params->approximation = true;
    return params;
  } else {
    params->approximation = stack.at(approx_index).to<std::string>() == "tanh";
    return params;
  }
}

std::shared_ptr<void> FillGeluFwdParams(const at::Stack& stack, size_t& size) {
  return FillGeluParams(stack, size, 1 /*Approximation Index in Fwd pass*/);
}

std::shared_ptr<void> FillGeluBwdParams(const at::Stack& stack, size_t& size) {
  return FillGeluParams(stack, size, 2 /*Approximation Index in Bwd pass*/);
}

std::vector<synapse_helpers::tensor> GeluCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_result_index = c10::nullopt) {
  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("gelu_fwd", op->ScalarType()),
       std::move(input),
       {{outshape, op->ScalarType(), final_result_index},
        {outshape, op->ScalarType()}},
       params.get(),
       size});
}

void Gelu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  size_t size = 0;
  auto params = FillGeluFwdParams(stack, size);
  auto Gelu =
      GeluCommonFunc(this, graph, {syn_in(0)}, outshape, params, size, 0);
  syn_out(0) = std::move(Gelu[0]);
}

} // namespace habana
