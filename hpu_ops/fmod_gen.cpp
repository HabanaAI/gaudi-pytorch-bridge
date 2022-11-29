/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/fmod.h"

namespace habana {

void Fmod::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& result_type = ScalarType();

  if (c10::isFloatingType(result_type)) {
    return OpBackend::AddNode(graph, stack);
  }
  // TPC mod_fwd supports only floating types. For integral types, use div_mod's
  // remainder output
  const auto outshape = BinaryOutputShape(stack)[0];
  auto mod = BuildOp(
      graph,
      "div_mod_" + habana_helpers::name_suffix_from_type(result_type),
      {syn_in(0), syn_in(1)},
      {{outshape, result_type}, {outshape, result_type, 0}});
  syn_out(0) = std::move(mod[1]);
}
} // namespace habana
