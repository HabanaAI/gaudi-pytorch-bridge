/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <c10/util/MathConstants.h>
#include "generated/hpu_op.h"
#include "hpu_op_helper.h"

namespace habana {

void Sinc::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto const_pi = ConstantHelper(graph, c10::pi<float>, ScalarType(), outshape);

  auto const_zero = ConstantHelper(graph, 0.0, ScalarType(), outshape);

  auto const_one = ConstantHelper(graph, 1.0, ScalarType(), outshape);

  const at::ScalarType& result_type = c10::ScalarType::Bool;
  auto mask = BuildOp(
      graph,
      "equal_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), const_zero.get()},
      {{outshape, result_type}});

  auto value = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), const_pi.get()},
      {{outshape, ScalarType()}});

  auto sine = BuildOp(
      graph,
      "sin_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {value[0].get()},
      {{outshape, ScalarType()}});

  auto reciprocal = BuildOp(
      graph,
      "reciprocal_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {value[0].get()},
      {{outshape, ScalarType()}});

  auto prod = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {sine[0].get(), reciprocal[0].get()},
      {{outshape, ScalarType()}});

  auto out = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask[0].get(), const_one.get(), prod[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  // output of where_outer is the output of this op
  syn_out(0) = std::move(out[0]);
}
} // namespace habana
