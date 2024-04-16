/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/pow.h"

namespace habana {

void PowOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto outshape = self.sizes();
  auto other = stack[1].toScalar();

  auto scalar_type = ScalarType();

  std::optional<synapse_helpers::tensor> cast{};
  if (scalar_type == c10::kChar) {
    cast = BuildCast(
        this, graph, syn_in(0), self.sizes().vec(), scalar_type, c10::kShort);
    scalar_type = c10::kShort;
  }

  if (other.toFloat() == 2.) {
    auto result = BuildOp(
        graph,
        get_guid_with_precision("mult_fwd", scalar_type),
        {cast.has_value() ? cast.value().get() : syn_in(0),
         cast.has_value() ? cast.value().get() : syn_in(0)},
        {{outshape, scalar_type, 0}});

    syn_out(0) = std::move(result[0]);
  } else if (other.toFloat() == 3.) {
    auto temp = BuildOp(
        graph,
        get_guid_with_precision("mult_fwd", scalar_type),
        {cast.has_value() ? cast.value().get() : syn_in(0),
         cast.has_value() ? cast.value().get() : syn_in(0)},
        {{outshape, scalar_type}});
    auto result = BuildOp(
        graph,
        get_guid_with_precision("mult_fwd", scalar_type),
        {temp[0].get(), cast.has_value() ? cast.value().get() : syn_in(0)},
        {{outshape, scalar_type, 0}});
    syn_out(0) = std::move(result[0]);

  } else if (other.toFloat() == 1.) {
    auto result = BuildOp(
        graph,
        "identity",
        {cast.has_value() ? cast.value().get() : syn_in(0)},
        {{outshape, scalar_type, 0}});
    syn_out(0) = std::move(result[0]);
  } else {
    auto result = BuildOp(
        graph,
        guid_,
        {cast.has_value() ? cast.value().get() : syn_in(0), syn_in(1)},
        {{outshape, scalar_type, 0}});
    syn_out(0) = std::move(result[0]);
  }
}

} // namespace habana
