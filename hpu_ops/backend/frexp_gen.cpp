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

#include "generated/backend/frexp.h"

namespace habana {

OutputMetaDataVector FrexpMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  const auto shape = self.sizes().vec();

  OutputMetaData mantissaMeta, exponentMeta;
  mantissaMeta.shape = exponentMeta.shape = shape;

  mantissaMeta.dtype = self.scalar_type();
  exponentMeta.dtype = c10::ScalarType::Int;

  return {mantissaMeta, exponentMeta};
}

c10::ScalarType GetKernelExponentType(const c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::BFloat16:
    case c10::ScalarType::Half:
      return c10::ScalarType::Short;
    default:
      return c10::ScalarType::Int;
  }
}

SharedMetaDataVector FrexpSharedMeta(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  auto inputType = input.scalar_type();
  auto rank = input.dim();

  if (c10::isIntegralType(inputType, true))
    inputType = c10::ScalarType::Float;

  auto exponentType = GetKernelExponentType(inputType);

  SharedMetaData frexpSharedMeta{"frexp"};
  frexpSharedMeta.inputs_data.emplace_back(rank, inputType);
  frexpSharedMeta.outputs_data = {{rank, exponentType}, {rank, inputType}};
  return {frexpSharedMeta};
}

void Frexp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto meta = FrexpMeta(stack);
  const auto kernelExponentType = GetKernelExponentType(meta[0].dtype);
  const bool castIsNeededForExponent = kernelExponentType != meta[1].dtype;
  const c10::optional<int> exponentFinalResultIndex =
      castIsNeededForExponent ? c10::nullopt : c10::optional<int>{1};

  auto frexp = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{meta[1].shape, kernelExponentType, exponentFinalResultIndex},
       {meta[0].shape, meta[0].dtype, 0}});

  syn_out(0) = std::move(frexp[1]);
  if (castIsNeededForExponent) {
    syn_out(1) = BuildCast(
        this,
        graph,
        frexp[0].get(),
        meta[1].shape,
        kernelExponentType,
        meta[1].dtype,
        1);
  } else {
    syn_out(1) = std::move(frexp[0]);
  }
}

} // namespace habana
