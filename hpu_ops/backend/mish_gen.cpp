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
      get_guid_with_precision("sigmoid_fwd", ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType()}});

  auto softplus_out = BuildOp(
      graph,
      get_guid_with_precision("softplus_fwd", ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType()}});

  auto tanh_out = BuildOp(
      graph,
      get_guid_with_precision("tanh_fwd", ScalarType()),
      {softplus_out[0].get()},
      {{outshape, ScalarType()}});

  auto mul_out1 = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {syn_in(1), sigmoid_out[0].get()},
      {{outshape, ScalarType()}});

  auto sq_out = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {tanh_out[0].get(), tanh_out[0].get()},
      {{outshape, ScalarType()}});

  auto mul_out2 = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {mul_out1[0].get(), sq_out[0].get()},
      {{outshape, ScalarType()}});

  auto sub_out = BuildOp(
      graph,
      get_guid_with_precision("sub", ScalarType()),
      {mul_out1[0].get(), mul_out2[0].get()},
      {{outshape, ScalarType()}});

  auto add_out = BuildOp(
      graph,
      get_guid_with_precision("add", ScalarType()),
      {tanh_out[0].get(), sub_out[0].get()},
      {{outshape, ScalarType()}});

  auto grad_input = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {syn_in(0), add_out[0].get()},
      {{outshape, ScalarType(), 0}});
  syn_out(0) = std::move(grad_input[0]);
}
} // namespace habana
