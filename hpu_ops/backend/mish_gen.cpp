/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/mish.h"
#include "generated/backend/mish_backward.h"
namespace habana {
void Mishbackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto sigmoid_out = BuildOp(
      graph,
      "sigmoid_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType()}});

  auto softplus_out = BuildOp(
      graph,
      "softplus_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType()}});

  auto tanh_out = BuildOp(
      graph,
      "tanh_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {softplus_out[0].get()},
      {{outshape, ScalarType()}});

  auto mul_out1 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1), sigmoid_out[0].get()},
      {{outshape, ScalarType()}});

  auto sq_out = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {tanh_out[0].get(), tanh_out[0].get()},
      {{outshape, ScalarType()}});

  auto mul_out2 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {mul_out1[0].get(), sq_out[0].get()},
      {{outshape, ScalarType()}});

  auto sub_out = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mul_out1[0].get(), mul_out2[0].get()},
      {{outshape, ScalarType()}});

  auto add_out = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {tanh_out[0].get(), sub_out[0].get()},
      {{outshape, ScalarType()}});

  auto grad_input = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), add_out[0].get()},
      {{outshape, ScalarType(), 0}});
  syn_out(0) = std::move(grad_input[0]);
}
} // namespace habana
