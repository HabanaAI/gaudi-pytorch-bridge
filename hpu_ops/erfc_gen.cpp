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
#include "hpu_op_helper.h"

namespace habana {
std::shared_ptr<void> FillErfcParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_ConstantKernel::Params);
  if (stack[0].toTensor().scalar_type() == c10::ScalarType::Int) {
    params->constant.i = 1;
  } else {
    params->constant.f = 1.0;
  }
  size = sizeof(params);
  return params;
}

void Erfc::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto erf = BuildOp(
      graph,
      "erf_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), false}});

  size_t size = 0;
  const auto& params = FillErfcParams(stack, size);

  auto constant = BuildOp(
      graph,
      "constant_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {},
      {{outshape, ScalarType(), false}},
      params.get(),
      size);

  auto erfc = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {constant[0].get(), erf[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  syn_out(0) = std::move(erfc[0]);
}
} // namespace habana
