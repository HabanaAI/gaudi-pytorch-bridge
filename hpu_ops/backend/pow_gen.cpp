/******************************************************************************
 * Copyright (C) 2021-2024 HabanaLabs, Ltd.
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

  if (other.toFloat() == 2.) {
    // fast path for square, mult_fwd_u8/s8_trunc will be used to handle
    // overflow
    auto result = BuildOp(
        graph,
        get_guid_with_precision("mult_fwd", scalar_type),
        {syn_in(0), syn_in(0)},
        {{outshape, scalar_type, 0}});
    syn_out(0) = std::move(result[0]);
  } else if (other.toFloat() == 1.) {
    // fast path for identity
    auto result =
        BuildOp(graph, "identity", {syn_in(0)}, {{outshape, scalar_type, 0}});
    syn_out(0) = std::move(result[0]);
  } else {
    // integer input will be cast to float and then the output will be trunced
    // if the scalar_type is u8/s8
    auto guid = guid_;
    std::optional<synapse_helpers::tensor> castedInput = c10::nullopt;
    auto compute_type = c10::isIntegralType(scalar_type, true)
        ? c10::ScalarType::Float
        : scalar_type;
    const bool is_cast_not_required =
        habana_helpers::getInternalDtype(compute_type) ==
        habana_helpers::getInternalDtype(scalar_type);
    NodeAttr::NodeOutputAttr out_attr = {outshape, compute_type};
    if (!is_cast_not_required) {
      castedInput = BuildCast(
          this,
          graph,
          syn_in(0),
          self.sizes().vec(),
          scalar_type,
          compute_type);
      guid = update_guid_dtype(guid_, compute_type);
    } else {
      out_attr.final_result_index = 0;
    }
    auto result = BuildOp(
        graph,
        guid,
        {castedInput.has_value() ? castedInput.value().get() : syn_in(0),
         syn_in(1)},
        {out_attr});
    if (is_cast_not_required) {
      syn_out(0) = std::move(result[0]);
    } else {
      auto castOut = OpBackend::BuildCast(
          this, graph, result[0].get(), outshape, compute_type, scalar_type, 0);
      syn_out(0) = std::move(castOut);
    }
  }
}

} // namespace habana
